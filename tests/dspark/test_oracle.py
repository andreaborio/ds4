#!/usr/bin/env python3
"""Model-free tests for the development-only DSpark numerical oracle."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import inspect
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.dspark_oracle import (  # noqa: E402
    DSPARK_RAW_CACHE_WIDTH,
    DSPARK_RAW_CACHE_WINDOW,
    DSPARK_STAGE_COUNT,
    DSPARK_TARGET_LAYER_IDS,
    MetadataError,
    PHYSICAL_HIDDEN_WIDTH,
    PHYSICAL_MODEL_ALIGNMENT,
    PHYSICAL_OUTPUT_RANK,
    PHYSICAL_Q_RANK,
    append_raw_cache,
    build_physical_stage_zero_fixture,
    capture_target_hidden_rows,
    commit_raw_cache_transaction,
    concatenate_target_captures,
    conditional_confidence,
    direct_stage_context_kv,
    dspark_attention_official,
    empty_raw_cache,
    finalize_draft_head,
    hc_head,
    hc_post,
    hc_pre,
    logical_raw_cache,
    main_project_and_norm,
    markov_greedy_draft,
    markov_sampled_draft,
    post_layer_hc_mean,
    pack_q8_0,
    payload_manifest,
    prefill_raw_cache,
    prepare_stage_zero,
    proposal_token_layout,
    proposal_raw_cache_view,
    rms_norm,
    run_synthetic_stage_chain,
    speculative_sample_exact,
    stage_zero_attention_half,
    unpack_q8_0,
    validate_0731_metadata,
)
from tools.dspark_oracle import mlx_optional  # noqa: E402
from tools.dspark_oracle import reference as dspark_reference  # noqa: E402
from tools.dspark_oracle.physical_fixture import (  # noqa: E402
    FFN_SELECTED_EXPERTS,
    build_physical_ffn_fixture,
    ffn_payload_manifest,
    unpack_iq2_xxs,
    unpack_q2_k,
)
from tools.dspark_oracle.support_schema import (  # noqa: E402
    SupportHeader,
    SupportSchemaError,
    TensorSpec,
    expected_tensor_schema,
    read_support_header,
    validate_support_file,
    validate_support_header,
    validate_support_layout,
)


FIXTURE_PATH = Path(__file__).with_name("fixtures-v1.json")
PROVENANCE_PATH = ROOT / "tools" / "dspark_oracle" / "provenance.json"
GENERATOR_PATH = ROOT / "tools" / "dspark_oracle" / "generate_fixtures.py"
DS4_SOURCE_PATH = ROOT / "ds4.c"
DS4_METAL_SOURCE_PATH = ROOT / "ds4_metal.m"
DSPARK_GRAPH_PATH = ROOT / "runtime" / "ds4_dspark_graph.inc"


_C_NONCODE_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*.*?\*/',
    re.DOTALL,
)
_C_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+)(?:[uUlL]+)?|"
    r"->|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|"
    r"[{}()\[\],;.=+\-!*/&|?:<>]"
)


def _strip_c_noncode(source: str) -> str:
    """Remove comments and literals without moving structural delimiters."""

    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group())

    return _C_NONCODE_RE.sub(replace, source)


def _matching_character(source: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    for index in range(start, len(source)):
        if source[index] == opener:
            depth += 1
        elif source[index] == closer:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unbalanced {opener}{closer} starting at byte {start}")


def _c_function_body(source: str, name: str) -> str:
    stripped = _strip_c_noncode(source)
    signature = re.compile(rf"\b{re.escape(name)}\s*\(")
    for match in signature.finditer(stripped):
        open_paren = stripped.find("(", match.start(), match.end())
        close_paren = _matching_character(stripped, open_paren, "(", ")")
        body_start = close_paren + 1
        while body_start < len(stripped) and stripped[body_start].isspace():
            body_start += 1
        if body_start >= len(stripped) or stripped[body_start] != "{":
            continue
        body_end = _matching_character(stripped, body_start, "{", "}")
        return stripped[body_start + 1:body_end]
    raise AssertionError(f"C function definition not found: {name}")


def _c_tokens(source: str) -> list[str]:
    return _C_TOKEN_RE.findall(source)


def _matching_token(
        tokens: list[str], start: int, opener: str, closer: str) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index] == opener:
            depth += 1
        elif tokens[index] == closer:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unbalanced token pair {opener}{closer} at {start}")


def _c_calls(
        tokens: list[str], name: str,
) -> list[tuple[int, int, list[list[str]]]]:
    calls: list[tuple[int, int, list[list[str]]]] = []
    for start in range(len(tokens) - 1):
        if tokens[start] != name or tokens[start + 1] != "(":
            continue
        end = _matching_token(tokens, start + 1, "(", ")")
        arguments: list[list[str]] = []
        argument: list[str] = []
        nesting: list[str] = []
        matching = {")": "(", "]": "[", "}": "{"}
        for token in tokens[start + 2:end]:
            if token in ("(", "[", "{"):
                nesting.append(token)
            elif token in matching:
                if not nesting or nesting.pop() != matching[token]:
                    raise AssertionError(f"unbalanced call arguments for {name}")
            if token == "," and not nesting:
                arguments.append(argument)
                argument = []
            else:
                argument.append(token)
        if argument or arguments:
            arguments.append(argument)
        calls.append((start, end, arguments))
    return calls


def _token_sequence_index(
        tokens: list[str], sequence: list[str], *, start: int = 0,
        end: int | None = None) -> int:
    if end is None:
        end = len(tokens)
    limit = end - len(sequence) + 1
    for index in range(start, max(start, limit)):
        if tokens[index:index + len(sequence)] == sequence:
            return index
    return -1


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_metadata_value(value: object) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", 8) + _gguf_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return struct.pack("<II", 4, value)
    if isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value):
        return (
            struct.pack("<IIQ", 9, 4, len(value))
            + b"".join(struct.pack("<I", item) for item in value)
        )
    raise AssertionError(f"unsupported synthetic metadata value: {value!r}")


def _synthetic_support_gguf(
        metadata: dict[str, object], tensors: dict[str, TensorSpec],
        *, version: int = 3) -> bytes:
    parts = [
        b"GGUF",
        struct.pack("<IQQ", version, len(tensors), len(metadata)),
    ]
    for key, value in metadata.items():
        parts.extend((_gguf_string(key), _gguf_metadata_value(value)))
    for name, spec in tensors.items():
        parts.extend((
            _gguf_string(name),
            struct.pack("<I", len(spec.dimensions)),
            struct.pack("<" + "Q" * len(spec.dimensions), *spec.dimensions),
            struct.pack("<IQ", spec.ggml_type, 0),
        ))
    return b"".join(parts)


class OracleFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_markov_greedy_fixture(self) -> None:
        case = self.fixture["cases"]["markovGreedy"]
        tokens, logits = markov_greedy_draft(
            np.asarray(case["baseLogits"], dtype=np.float64),
            case["firstPreviousToken"],
            np.asarray(case["embedding"], dtype=np.float64),
            np.asarray(case["projection"], dtype=np.float64),
        )
        self.assertEqual(tokens.tolist(), case["expected"]["tokens"])
        np.testing.assert_allclose(
            logits,
            np.asarray(case["expected"]["stepLogits"]),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_markov_sampled_fixture_is_sequential_and_reproducible(self) -> None:
        case = self.fixture["cases"]["markovSampled"]
        result = markov_sampled_draft(
            np.asarray(case["baseLogits"], dtype=np.float64),
            case["firstPreviousToken"],
            np.asarray(case["embedding"], dtype=np.float64),
            np.asarray(case["projection"], dtype=np.float64),
            case["samplingUniforms"],
            temperature=case["temperature"],
        )
        self.assertEqual(result.tokens.tolist(), case["expected"]["tokens"])
        np.testing.assert_allclose(
            result.corrected_logits,
            case["expected"]["correctedLogits"],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.probabilities,
            case["expected"]["probabilities"],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.sum(result.probabilities, axis=1),
            [1.0, 1.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )

    @staticmethod
    def _hc_capture_input(case: dict[str, object]) -> np.ndarray:
        shape = tuple(int(value) for value in case["shape"])
        generator = case["generator"]
        token = np.arange(shape[0], dtype=np.float64)[:, None, None]
        hc = np.arange(shape[1], dtype=np.float64)[None, :, None]
        dim = np.arange(shape[2], dtype=np.int64)[None, None, :]
        dim_term = (
            (dim % int(generator["dimensionPeriod"]))
            + int(generator["dimensionOffset"])
        ) * float(generator["dimensionScale"])
        return (
            token * float(generator["tokenScale"])
            + hc * float(generator["hcScale"])
            + dim_term
        )

    def test_post_layer_hc_mean_realistic_fixture(self) -> None:
        case = self.fixture["cases"]["postLayerHCMean"]
        hidden = self._hc_capture_input(case)
        captured = post_layer_hc_mean(hidden)
        self.assertEqual(list(hidden.shape), case["shape"])
        self.assertEqual(list(captured.shape), case["expectedShape"])
        self.assertFalse(np.shares_memory(hidden, captured))
        for sample in case["expectedSamples"]:
            token, dim = sample["index"]
            self.assertEqual(captured[token, dim], sample["value"])

    def test_post_layer_hc_mean_rejects_shape_aliases_and_undersize(self) -> None:
        valid = np.zeros((1, 4, 4096), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, r"\[token, 4, 4096\]"):
            post_layer_hc_mean(valid.reshape(4, 4096))
        with self.assertRaisesRegex(ValueError, r"\[token, 4, 4096\]"):
            post_layer_hc_mean(valid[:, :3, :])
        with self.assertRaisesRegex(ValueError, r"\[token, 4, 4096\]"):
            post_layer_hc_mean(valid[:, :, :-1])

    @staticmethod
    def _target_capture_input(
        case: dict[str, object], token_count: int
    ) -> np.ndarray:
        hc_lanes, hidden_width = (
            int(value) for value in case["shapePerLayer"]
        )
        generator = case["generator"]
        layer = np.arange(3, dtype=np.float64)[:, None, None, None]
        token = np.arange(token_count, dtype=np.float64)[None, :, None, None]
        hc = np.arange(hc_lanes, dtype=np.float64)[None, None, :, None]
        dimension = np.arange(hidden_width, dtype=np.int64)[None, None, None, :]
        dimension_term = (
            (dimension % int(generator["dimensionPeriod"]))
            + int(generator["dimensionOffset"])
        ) * float(generator["dimensionScale"])
        return (
            layer * float(generator["layerScale"])
            + token * float(generator["tokenScale"])
            + hc * float(generator["hcScale"])
            + dimension_term
        )

    @staticmethod
    def _assert_capture_samples(
        rows: np.ndarray, samples: list[dict[str, object]]
    ) -> None:
        for sample in samples:
            actual = rows[int(sample["layer"]), int(sample["dimension"])]
            assert actual == float(sample["value"]), sample

    def test_target_capture_decode_rows_are_ordered_and_non_degenerate(self) -> None:
        case = self.fixture["cases"]["targetCaptureRows"]
        hidden = self._target_capture_input(case, case["decodeTokenCount"])
        result = capture_target_hidden_rows(
            tuple(hidden), case["layerIds"], phase="decode",
            start_position=case["decodeStartPosition"],
        )
        self.assertEqual(result.layer_ids, DSPARK_TARGET_LAYER_IDS)
        self.assertEqual(result.phase, "decode")
        self.assertEqual(
            result.token_index, case["expected"]["decodeTokenIndex"]
        )
        self.assertEqual(
            result.absolute_token_position,
            case["expected"]["decodeAbsoluteTokenPosition"],
        )
        self.assertEqual(result.rows.shape, (3, 4096))
        self.assertEqual(result.history_rows.shape, (1, 3, 4096))
        np.testing.assert_array_equal(result.history_rows[0], result.rows)
        self._assert_capture_samples(
            result.rows, case["expected"]["decodeSamples"]
        )
        self.assertFalse(np.shares_memory(result.rows, hidden))
        self.assertFalse(np.array_equal(result.rows[0], result.rows[1]))
        self.assertFalse(np.array_equal(result.rows[1], result.rows[2]))

        # The lanes are intentionally distinct: selecting one lane instead of
        # the post-HC mean cannot accidentally satisfy the closed samples.
        self.assertFalse(np.array_equal(result.rows, hidden[:, 0, 0, :]))

    def test_target_capture_prefill_keeps_last_128_and_distinct_frontier(self) -> None:
        case = self.fixture["cases"]["targetCaptureRows"]
        hidden = self._target_capture_input(case, case["prefillTokenCount"])
        result = capture_target_hidden_rows(
            tuple(hidden), case["layerIds"], phase="prefill",
            start_position=case["prefillStartPosition"],
        )
        self.assertEqual(result.phase, "prefill")
        self.assertEqual(
            result.token_index, case["expected"]["prefillTokenIndex"]
        )
        self.assertEqual(
            result.absolute_token_position,
            case["expected"]["prefillAbsoluteTokenPosition"],
        )
        self.assertEqual(
            result.history_token_start, case["expected"]["historyTokenStart"]
        )
        self.assertEqual(
            result.history_rows.shape,
            (case["expected"]["historyLength"], 3, 4096),
        )
        self._assert_capture_samples(
            result.rows, case["expected"]["prefillSamples"]
        )
        for sample in case["expected"]["historySamples"]:
            actual = result.history_rows[
                int(sample["logical"]),
                int(sample["layer"]),
                int(sample["dimension"]),
            ]
            self.assertEqual(actual, sample["value"])
        np.testing.assert_array_equal(result.history_rows[-1], result.rows)
        self.assertFalse(np.array_equal(result.history_rows[0], result.rows))

    def test_official_pending_token_and_candidate_position_shift(self) -> None:
        case = self.fixture["cases"]["proposalTokenLayout"]
        result = proposal_token_layout(
            case["lastTargetPosition"],
            case["pendingTokenId"],
            case["noiseTokenId"],
            block_size=case["blockSize"],
        )
        self.assertEqual(result.pending_token_id, case["pendingTokenId"])
        self.assertEqual(
            result.input_token_ids.tolist(), case["expected"]["inputTokenIds"]
        )
        self.assertEqual(
            result.input_positions.tolist(), case["expected"]["inputPositions"]
        )
        self.assertEqual(
            result.proposed_output_positions.tolist(),
            case["expected"]["proposedOutputPositions"],
        )
        self.assertNotIn(
            case["lastTargetPosition"], result.input_positions.tolist()
        )
        self.assertEqual(
            result.proposed_output_positions[0], result.input_positions[0] + 1
        )

    def test_target_capture_rejects_layer_and_phase_mutations(self) -> None:
        case = self.fixture["cases"]["targetCaptureRows"]
        decode = self._target_capture_input(case, case["decodeTokenCount"])
        prefill = self._target_capture_input(case, case["prefillTokenCount"])
        with self.assertRaisesRegex(ValueError, "ordered 40, 41, 42"):
            capture_target_hidden_rows(
                tuple(decode[[1, 0, 2]]), [41, 40, 42], phase="decode"
            )
        with self.assertRaisesRegex(ValueError, "ordered 40, 41, 42"):
            capture_target_hidden_rows(
                tuple(decode), [40, 40, 42], phase="decode"
            )
        with self.assertRaisesRegex(ValueError, "exactly one token row"):
            capture_target_hidden_rows(
                tuple(prefill), case["layerIds"], phase="decode"
            )
        with self.assertRaisesRegex(ValueError, "same token count"):
            capture_target_hidden_rows(
                (decode[0], prefill[1], decode[2]),
                case["layerIds"],
                phase="prefill",
            )
        with self.assertRaisesRegex(ValueError, "'decode' or 'prefill'"):
            capture_target_hidden_rows(
                tuple(decode), case["layerIds"], phase="verify"
            )

    @staticmethod
    def _fixture_matrix(description: object) -> np.ndarray:
        if isinstance(description, dict):
            shape = tuple(int(item) for item in description["shape"])
            kind = description.get("kind")
            if "fill" in description:
                return np.full(shape, float(description["fill"]),
                               dtype=np.float64)
            if kind == "denseModular":
                rows, columns = shape
                result = np.empty(shape, dtype=np.float64)
                for row in range(rows):
                    for column in range(columns):
                        raw = (
                            row * int(description["rowMultiplier"])
                            + column * int(description["columnMultiplier"])
                        ) % int(description["modulus"]) - int(
                            description["offset"]
                        )
                        replacement = float(description["zeroReplacement"])
                        result[row, column] = (
                            replacement if raw == 0 else raw
                        ) / float(description["divisor"])
                return result
            if kind == "periodicModular":
                if len(shape) != 1:
                    raise AssertionError("periodicModular must be a vector")
                result = np.empty(shape, dtype=np.float64)
                for index in range(shape[0]):
                    raw = (
                        index * int(description["indexMultiplier"])
                    ) % int(description["modulus"]) - int(
                        description["offset"]
                    )
                    replacement = float(description["zeroReplacement"])
                    result[index] = (
                        replacement if raw == 0 else raw
                    ) / float(description["divisor"])
                return result
            raise AssertionError(f"unknown fixture matrix generator: {kind}")
        return np.asarray(description, dtype=np.float64)

    @staticmethod
    def _array_digest(array: np.ndarray, dtype: str) -> str:
        payload = np.asarray(array, dtype=np.dtype(dtype)).tobytes(order="C")
        return hashlib.sha256(payload).hexdigest()

    def _assert_frozen_samples(
        self, array: np.ndarray, samples: list[dict[str, object]]
    ) -> None:
        for sample in samples:
            self.assertEqual(
                array[int(sample["row"]), int(sample["dimension"])],
                sample["value"],
                sample,
            )

    def _assert_bfloat16_boundary(self, array: np.ndarray) -> None:
        bits = np.asarray(array, dtype=np.float32).view(np.uint32)
        self.assertTrue(np.all((bits & np.uint32(0xFFFF)) == 0))

    def test_stage_zero_main_projection_norm_and_noise_layout(self) -> None:
        case = self.fixture["cases"]["stageSetup"]
        target = np.asarray(case["targetHidden"], dtype=np.float64)
        concatenated = concatenate_target_captures(target)
        np.testing.assert_array_equal(
            concatenated, case["expected"]["concatenated"]
        )
        self.assertFalse(np.shares_memory(concatenated, target))
        np.testing.assert_array_equal(
            concatenated @ np.asarray(case["mainProjection"]).T,
            case["expected"]["preNorm"],
        )
        projected = main_project_and_norm(
            target,
            np.asarray(case["mainProjection"], dtype=np.float64),
            case["mainNormWeight"],
        )
        setup = prepare_stage_zero(
            target,
            case["pendingEmbedding"],
            case["noiseEmbedding"],
            np.asarray(case["mainProjection"], dtype=np.float64),
            case["mainNormWeight"],
            block_size=case["blockSize"],
            hc_lanes=case["hcLanes"],
        )
        expected_main = np.asarray(case["expected"]["mainHidden"])
        np.testing.assert_allclose(projected, expected_main, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(setup.main_hidden, expected_main,
                                   rtol=0.0, atol=1e-14)
        self.assertEqual(setup.draft_hidden.shape, (5, 4, 2))
        for lane in range(4):
            np.testing.assert_array_equal(
                setup.draft_hidden[:, lane, :],
                np.asarray(case["expected"]["draftRows"]),
            )

        # Every target tap participates with a stage-distinct coefficient.
        # Swapping any pair changes both the concatenation and the projected
        # stage-zero main row.
        for first, second in ((0, 1), (1, 2), (0, 2)):
            mutated = np.array(target, copy=True)
            mutated[[first, second]] = mutated[[second, first]]
            self.assertFalse(np.array_equal(
                concatenate_target_captures(mutated), concatenated
            ))
            self.assertFalse(np.allclose(
                main_project_and_norm(
                    mutated,
                    np.asarray(case["mainProjection"], dtype=np.float64),
                    case["mainNormWeight"],
                ),
                projected,
            ))

    def test_main_projection_requires_three_captures(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            main_project_and_norm(
                np.zeros((2, 2)), np.zeros((2, 4)), np.ones(2)
            )

    def test_official_context_kv_is_direct_with_rope_and_nope_fp8(self) -> None:
        case = self.fixture["cases"]["directContextKV"]
        projection_shape = tuple(case["projectionGenerator"]["shape"])
        projection = np.zeros(projection_shape, dtype=np.float64)
        projection[
            np.arange(projection_shape[0]),
            np.arange(projection_shape[0]) % projection_shape[1],
        ] = 1.0
        result = direct_stage_context_kv(
            np.asarray(case["mainX"], dtype=np.float64),
            projection,
            self._fixture_matrix(case["normWeight"]),
            case["absolutePositions"],
            eps=case["normEps"],
            rope_theta=case["ropeTheta"],
        )
        expected = case["expected"]
        self.assertEqual(result.projected.shape, (2, 512))
        self.assertEqual(result.normalized.shape, (2, 512))
        self.assertEqual(result.stored.shape, (2, 512))
        self.assertEqual(result.nonrope_scales.shape, (2, 7))
        self.assertEqual(
            result.absolute_positions.tolist(), case["absolutePositions"]
        )
        for field, samples in (
            ("projected", expected["projectedSamples"]),
            ("normalized", expected["normalizedSamples"]),
            ("roped", expected["ropedSamples"]),
            ("stored", expected["storedSamples"]),
        ):
            array = getattr(result, field)
            for sample in samples:
                self.assertEqual(
                    array[int(sample["token"]), int(sample["dimension"])],
                    sample["value"],
                )
        np.testing.assert_array_equal(
            result.nonrope_scales, expected["nonropeScales"]
        )
        # Position zero is identity RoPE.  Position 129 must rotate only the
        # tail; the in-place FP8 Q/DQ must alter only the 448 non-RoPE values.
        np.testing.assert_array_equal(result.normalized[0, -64:],
                                      result.roped[0, -64:])
        self.assertFalse(np.array_equal(result.normalized[1, -64:],
                                        result.roped[1, -64:]))
        np.testing.assert_array_equal(result.roped[:, -64:],
                                      result.stored[:, -64:])
        self.assertFalse(np.array_equal(result.roped[:, :-64],
                                        result.stored[:, :-64]))

        zero_result = direct_stage_context_kv(
            np.zeros((1, 4), dtype=np.float64),
            projection,
            self._fixture_matrix(case["normWeight"]),
            [0],
            eps=case["normEps"],
            rope_theta=case["ropeTheta"],
        )
        np.testing.assert_array_equal(
            zero_result.nonrope_scales,
            np.full((1, 7), 2.0 ** -22, dtype=np.float64),
        )
        np.testing.assert_array_equal(zero_result.stored,
                                      np.zeros((1, 512)))

    def test_raw_context_finalizer_freezes_every_precision_boundary(self) -> None:
        self.assertEqual(self.fixture["schemaVersion"], 4)
        case = self.fixture["cases"]["rawContextFinalizer"]
        contract = case["rowContract"]
        self.assertEqual(contract["candidateBlockSize"], 5)
        self.assertEqual(contract["captureRows"], 6)
        self.assertEqual(contract["candidateRowIndices"], [0, 1, 2, 3, 4])
        self.assertEqual(contract["verifierOnlyRowIndex"], 5)

        target = np.asarray(case["targetHidden"], dtype=np.float64)
        main_projection = np.asarray(
            case["mainProjection"], dtype=np.float64
        )
        context_projection = self._fixture_matrix(
            case["contextProjectionGenerator"]
        )
        context_norm = self._fixture_matrix(
            case["contextNormWeightGenerator"]
        )
        packed = concatenate_target_captures(target)
        main_x = main_project_and_norm(
            target, main_projection, case["mainNormWeight"],
            eps=case["normEps"],
        )
        expected = case["expected"]
        np.testing.assert_allclose(
            main_x, expected["mainX"], rtol=0.0, atol=1.0e-14
        )
        self.assertEqual(
            self._array_digest(packed, "<f8"),
            expected["digests"]["packedCapturesF64"],
        )
        self.assertEqual(
            self._array_digest(main_x, "<f8"),
            expected["digests"]["mainXF64"],
        )

        result = direct_stage_context_kv(
            main_x,
            context_projection,
            context_norm,
            case["absolutePositions"],
            eps=case["normEps"],
            rope_theta=case["ropeTheta"],
        )
        self.assertEqual(result.absolute_positions.tolist(),
                         case["absolutePositions"])
        for field, digest_name, samples_name in (
            ("projected", "projectedF32", "projectedSamples"),
            ("normalized", "normalizedF32", "normalizedSamples"),
            ("roped", "ropedF32", "ropedSamples"),
            ("stored", "storedF32", "storedSamples"),
        ):
            boundary = getattr(result, field)
            self.assertEqual(boundary.shape, (6, 512))
            self._assert_bfloat16_boundary(boundary)
            self.assertEqual(
                self._array_digest(boundary, "<f4"),
                expected["digests"][digest_name],
            )
            self._assert_frozen_samples(boundary, expected[samples_name])
        self.assertEqual(result.nonrope_scales.shape, (6, 7))
        np.testing.assert_array_equal(
            result.nonrope_scales, expected["nonropeScales"]
        )
        self.assertEqual(
            self._array_digest(result.nonrope_scales, "<f4"),
            expected["digests"]["nonropeScalesF32"],
        )

        # RoPE changes only the final 64 dimensions.  FP8 Q/DQ then changes
        # only the 448-wide prefix before the final BF16 store.
        np.testing.assert_array_equal(
            result.normalized[:, :448], result.roped[:, :448]
        )
        self.assertFalse(np.array_equal(
            result.normalized[1:, 448:], result.roped[1:, 448:]
        ))
        np.testing.assert_array_equal(
            result.roped[:, 448:], result.stored[:, 448:]
        )
        self.assertFalse(np.array_equal(
            result.roped[:, :448], result.stored[:, :448]
        ))
        mantissas, _ = np.frexp(result.nonrope_scales)
        np.testing.assert_array_equal(
            mantissas, np.full((6, 7), 0.5, dtype=np.float64)
        )

        # Every supported verifier/capture batch C=1..6 is row-prefix stable.
        # C=6 does not create a sixth DSpark candidate; the fixture contract
        # marks row index five as verifier-only above.
        for rows in range(1, 7):
            prefix_main = main_project_and_norm(
                target[:rows], main_projection, case["mainNormWeight"],
                eps=case["normEps"],
            )
            prefix = direct_stage_context_kv(
                prefix_main,
                context_projection,
                context_norm,
                case["absolutePositions"][:rows],
                eps=case["normEps"],
                rope_theta=case["ropeTheta"],
            )
            np.testing.assert_array_equal(prefix_main, main_x[:rows])
            for field in ("projected", "normalized", "roped", "stored"):
                np.testing.assert_array_equal(
                    getattr(prefix, field), getattr(result, field)[:rows]
                )
            np.testing.assert_array_equal(
                prefix.nonrope_scales, result.nonrope_scales[:rows]
            )

    def test_context_kv_rejects_candidate_or_position_aliases(self) -> None:
        main = np.ones((2, 4), dtype=np.float64)
        projection = np.ones((512, 4), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "match main_x"):
            direct_stage_context_kv(
                main, projection, np.ones(512), [9]
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            direct_stage_context_kv(
                main, projection, np.ones(512), [9, -1]
            )

    def test_hyper_connection_split_pre_post_and_head(self) -> None:
        case = self.fixture["cases"]["hyperConnection"]
        hidden = np.asarray(case["hidden"], dtype=np.float64)
        function = self._fixture_matrix(case["function"])
        reduced, split = hc_pre(
            hidden,
            function,
            case["scale"],
            case["base"],
            norm_eps=case["normEps"],
            hc_eps=case["hcEps"],
            iterations=case["iterations"],
        )
        np.testing.assert_allclose(split.pre, case["expected"]["pre"],
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(split.post, case["expected"]["post"],
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            split.combination,
            case["expected"]["combination"],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(reduced, case["expected"]["reduced"],
                                   rtol=0.0, atol=1e-14)
        expanded = hc_post(case["branchOutput"], hidden, split)
        np.testing.assert_allclose(
            expanded,
            np.asarray(case["expected"]["postOutput"]),
            rtol=0.0,
            atol=1e-14,
        )
        collapsed = hc_head(
            hidden,
            self._fixture_matrix(case["headFunction"]),
            case["headScale"],
            case["headBase"],
            norm_eps=case["normEps"],
            hc_eps=case["hcEps"],
        )
        np.testing.assert_allclose(collapsed, case["expected"]["head"],
                                   rtol=0.0, atol=1e-14)

        lane_swapped = hidden[[1, 0, 2, 3]]
        swapped_reduced, swapped_split = hc_pre(
            lane_swapped,
            function,
            case["scale"],
            case["base"],
            norm_eps=case["normEps"],
            hc_eps=case["hcEps"],
            iterations=case["iterations"],
        )
        self.assertFalse(np.allclose(swapped_reduced, reduced))
        self.assertFalse(np.allclose(swapped_split.pre, split.pre))
        wrong_flatten = hidden.T.reshape(4, 2)
        wrong_reduced, _ = hc_pre(
            wrong_flatten,
            function,
            case["scale"],
            case["base"],
            norm_eps=case["normEps"],
            hc_eps=case["hcEps"],
            iterations=case["iterations"],
        )
        self.assertFalse(np.allclose(wrong_reduced, reduced))
        with self.assertRaisesRegex(ValueError, r"\[24, 4 \* hidden\]"):
            hc_pre(
                hidden,
                function.T,
                case["scale"],
                case["base"],
                norm_eps=case["normEps"],
                hc_eps=case["hcEps"],
                iterations=case["iterations"],
            )

    def test_synthetic_three_stage_chain_is_ordered_and_shares_main(self) -> None:
        case = self.fixture["cases"]["stageChain"]
        main = np.asarray(case["mainHidden"], dtype=np.float64)
        main_before = np.array(main, copy=True)
        result = run_synthetic_stage_chain(
            np.asarray(case["draftHidden"], dtype=np.float64),
            main,
            np.asarray(case["stageWeights"], dtype=np.float64),
            np.asarray(case["mainWeights"], dtype=np.float64),
            np.asarray(case["stageBiases"], dtype=np.float64),
        )
        self.assertEqual(len(result.stage_outputs), DSPARK_STAGE_COUNT)
        for actual, expected in zip(
            result.stage_outputs, case["expectedStageOutputs"]
        ):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(main, main_before)

        # The matrices are deliberately non-commutative and stage-distinct.
        # Reversing or reusing a stage must not accidentally match the oracle.
        reversed_result = run_synthetic_stage_chain(
            np.asarray(case["draftHidden"], dtype=np.float64),
            main,
            np.asarray(case["stageWeights"], dtype=np.float64)[::-1],
            np.asarray(case["mainWeights"], dtype=np.float64)[::-1],
            np.asarray(case["stageBiases"], dtype=np.float64)[::-1],
        )
        self.assertFalse(np.allclose(
            reversed_result.stage_outputs[-1], result.stage_outputs[-1]
        ))
        reused_result = run_synthetic_stage_chain(
            np.asarray(case["draftHidden"], dtype=np.float64),
            main,
            np.repeat(
                np.asarray(case["stageWeights"], dtype=np.float64)[:1], 3, axis=0
            ),
            np.repeat(
                np.asarray(case["mainWeights"], dtype=np.float64)[:1], 3, axis=0
            ),
            np.repeat(
                np.asarray(case["stageBiases"], dtype=np.float64)[:1], 3, axis=0
            ),
        )
        self.assertFalse(np.allclose(
            reused_result.stage_outputs[-1], result.stage_outputs[-1]
        ))
        with self.assertRaisesRegex(ValueError, r"\[3, hidden, hidden\]"):
            run_synthetic_stage_chain(
                np.asarray(case["draftHidden"], dtype=np.float64),
                main,
                np.asarray(case["stageWeights"], dtype=np.float64)[:2],
                np.asarray(case["mainWeights"], dtype=np.float64)[:2],
                np.asarray(case["stageBiases"], dtype=np.float64)[:2],
            )

    @staticmethod
    def _raw_cache_rows(
        positions: np.ndarray, case: dict[str, object]
    ) -> np.ndarray:
        generator = case["rowGenerator"]
        position = positions.astype(np.float64)[:, None, None]
        stage = np.arange(case["stageCount"], dtype=np.float64)[None, :, None]
        dimension = np.arange(case["width"], dtype=np.float64)[None, None, :]
        return (
            stage * float(generator["stageScale"])
            + position * float(generator["positionScale"])
            + dimension * float(generator["dimensionScale"])
        )

    @staticmethod
    def _raw_cache_drafts(case: dict[str, object]) -> np.ndarray:
        generator = case["draftGenerator"]
        stage = np.arange(case["stageCount"], dtype=np.float64)[:, None, None]
        slot = np.arange(5, dtype=np.float64)[None, :, None]
        dimension = np.arange(case["width"], dtype=np.float64)[None, None, :]
        return (
            float(generator["base"])
            + stage * float(generator["stageScale"])
            + slot * float(generator["slotScale"])
            + dimension * float(generator["dimensionScale"])
        )

    @staticmethod
    def _dspark_attention_inputs() -> tuple[
        object, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Deterministic BF16-sensitive full-ring attention fixture."""

        generator = np.random.default_rng(0xD54A)
        rows = generator.standard_normal(
            (129, DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WIDTH),
            dtype=np.float32,
        ) * np.float32(0.35)
        state = prefill_raw_cache(rows, start_position=0)
        queries = generator.standard_normal(
            (5, 64, DSPARK_RAW_CACHE_WIDTH), dtype=np.float32
        ) * np.float32(0.35)
        drafts = generator.standard_normal(
            (DSPARK_STAGE_COUNT, 5, DSPARK_RAW_CACHE_WIDTH),
            dtype=np.float32,
        ) * np.float32(0.35)
        sinks = np.linspace(-1.75, 1.25, 64, dtype=np.float32)
        return state, queries, drafts, sinks

    @staticmethod
    def _stage_zero_attention_half_inputs(
        committed_count: int,
    ) -> dict[str, object]:
        """Compact weights around a final 5x64x512 attention seam."""

        if committed_count not in (2, 128):
            raise ValueError("fixture admits only C=2 or C=128")
        generator = np.random.default_rng(0x0731A770)
        hidden_width = 8
        q_rank = 8
        output_rank = 2
        hidden = generator.standard_normal(
            (5, 4, hidden_width), dtype=np.float32
        ) * np.float32(0.35)
        # HC weights are zero here so both oracles enter the first named BF16
        # boundary identically.  Non-degenerate HC mixes are covered by the
        # independent HC fixture; random dense weights here would compound a
        # one-ULP HC difference through every later boundary.
        hc_function = np.zeros((24, 4 * hidden_width), dtype=np.float32)
        attention_norm = generator.uniform(
            0.65, 1.35, size=hidden_width
        ).astype(np.float32)
        q_a = np.zeros((q_rank, hidden_width), dtype=np.float32)
        q_a[np.arange(q_rank), np.arange(hidden_width)] = np.asarray(
            [0.5, -0.75, 1.0, -0.25] * 2, dtype=np.float32
        )
        q_a_norm = np.arange(q_rank, dtype=np.float32) / 16.0 + 0.75
        q_b = np.zeros((64 * 512, q_rank), dtype=np.float32)
        q_b_view = q_b.reshape(64, 512, q_rank)
        dimensions = np.arange(512)
        for column in range(q_rank):
            q_b_view[:, :, column] = np.where(
                ((dimensions >> column) & 1) == 0, 0.0625, -0.0625
            )
        # One exact sign flip per head makes the compact Q-B fixture
        # head-distinguishing without perturbing the otherwise shared pattern.
        for head in range(64):
            q_b_view[head, head, head % q_rank] *= -1.0
        kv = np.zeros((512, hidden_width), dtype=np.float32)
        kv_rows = np.arange(kv.shape[0])
        # All five compact hidden rows have a positive lane 1.  Selecting it
        # keeps the synthetic V values away from cancellation around zero;
        # four exact positive scales preserve dimension diversity.
        kv[:, 1] = ((kv_rows % 4).astype(np.float32) + 1.0) / 4.0
        kv_norm = (
            (np.arange(512) % 11).astype(np.float32) + 8.0
        ) / 512.0
        ring_source = generator.standard_normal(
            (130, 3, 512), dtype=np.float32
        ) * np.float32(0.28)
        ring_source = (
            np.abs(ring_source) / np.float32(32.0)
            + np.float32(1.0 / 32.0)
        )
        state = prefill_raw_cache(
            ring_source[:2] if committed_count == 2 else ring_source,
            start_position=0,
        )
        other_drafts = generator.standard_normal(
            (3, 5, 512), dtype=np.float32
        ) * np.float32(0.24)
        sinks = np.linspace(-1.5, 1.125, 64, dtype=np.float32)
        output_a = np.zeros(
            (8, output_rank, 8 * 512), dtype=np.float32
        )
        output_a_values = np.asarray(
            [0.125, -0.0625, 0.25, -0.125,
             0.03125, -0.25, 0.0625, 0.125],
            dtype=np.float32,
        )
        for group in range(8):
            for rank in range(output_rank):
                rank_scale = 1.0 if rank == 0 else -0.5
                for index, value in enumerate(output_a_values):
                    dimension = (
                        group * 131 + rank * 277 + index * 503
                    ) % 4096
                    output_a[group, rank, dimension] = value * rank_scale
        output_b = np.fromfunction(
            lambda row, column: ((row * 5 + column * 3) % 9 - 4) / 8,
            (hidden_width, 8 * output_rank),
            dtype=int,
        ).astype(np.float32)
        positions = np.arange(
            state.token_start + state.length,
            state.token_start + state.length + 5,
            dtype=np.int64,
        )
        return {
            "hidden_input": hidden,
            "hc_function_weight": hc_function,
            "hc_scale": np.zeros(3, dtype=np.float32),
            "hc_base": np.zeros(24, dtype=np.float32),
            "attention_norm_weight": attention_norm,
            "q_a_weight": q_a,
            "q_a_norm_weight": q_a_norm,
            "q_b_weight": q_b,
            "kv_weight": kv,
            "kv_norm_weight": kv_norm,
            "absolute_positions": positions,
            "raw_cache": state,
            "other_stage_draft_rows": other_drafts,
            "attention_sinks": sinks,
            "output_a_weight": output_a,
            "output_b_weight": output_b,
        }

    @staticmethod
    def _round_fixture_bfloat16(value: np.ndarray) -> np.ndarray:
        source = np.asarray(value, dtype=np.float32)
        bits = np.array(source, copy=True).view(np.uint32)
        rounding = np.uint32(0x7FFF) + (
            (bits >> np.uint32(16)) & np.uint32(1)
        )
        return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)

    @staticmethod
    def _mlx_stage_zero_attention_half(
        inputs: dict[str, object],
        **kwargs: object,
    ) -> object:
        state = inputs["raw_cache"]
        return mlx_optional.stage_zero_attention_half(
            inputs["hidden_input"],
            inputs["hc_function_weight"],
            inputs["hc_scale"],
            inputs["hc_base"],
            inputs["attention_norm_weight"],
            inputs["q_a_weight"],
            inputs["q_a_norm_weight"],
            inputs["q_b_weight"],
            inputs["kv_weight"],
            inputs["kv_norm_weight"],
            inputs["absolute_positions"],
            state.rows[0],
            state.length,
            state.token_start,
            inputs["other_stage_draft_rows"],
            inputs["attention_sinks"],
            inputs["output_a_weight"],
            inputs["output_b_weight"],
            **kwargs,
        )

    @staticmethod
    def _assert_raw_samples(
        array: np.ndarray,
        samples: list[dict[str, object]],
        index_name: str,
    ) -> None:
        for sample in samples:
            actual = array[
                int(sample["stage"]),
                int(sample[index_name]),
                int(sample["dimension"]),
            ]
            assert actual == float(sample["value"]), sample

    def test_final_0731_raw_cache_boundaries_and_transactionality(self) -> None:
        case = self.fixture["cases"]["rawCache"]
        self.assertEqual(case["stageCount"], DSPARK_STAGE_COUNT)
        self.assertEqual(case["window"], DSPARK_RAW_CACHE_WINDOW)
        self.assertEqual(case["width"], DSPARK_RAW_CACHE_WIDTH)
        prefill = self._raw_cache_rows(
            np.arange(case["prefillTokenCount"], dtype=np.int64), case
        )
        state = prefill_raw_cache(
            prefill, start_position=case["startPosition"]
        )
        expected = case["expected"]
        self.assertEqual(state.token_start, expected["beforeWrapTokenStart"])
        self.assertEqual(state.length, expected["beforeWrapLength"])
        self._assert_raw_samples(
            logical_raw_cache(state), expected["beforeWrapSamples"], "logical"
        )

        row128 = self._raw_cache_rows(np.asarray([128]), case)[0]
        state128 = append_raw_cache(state, 128, row128)
        self.assertEqual(state128.token_start, expected["after128TokenStart"])
        logical128 = logical_raw_cache(state128)
        logical_samples = [
            item for item in expected["after128Samples"] if "logical" in item
        ]
        physical_samples = [
            item for item in expected["after128Samples"] if "physical" in item
        ]
        self._assert_raw_samples(logical128, logical_samples, "logical")
        self._assert_raw_samples(state128.rows, physical_samples, "physical")

        draft = self._raw_cache_drafts(case)
        before = np.array(state128.rows, copy=True)
        view = proposal_raw_cache_view(
            state128, case["proposalPosition"], draft
        )
        self.assertEqual(
            view.shape,
            (DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WINDOW + 5,
             DSPARK_RAW_CACHE_WIDTH),
        )
        self._assert_raw_samples(view, expected["proposalSamples"], "view")
        np.testing.assert_array_equal(state128.rows, before)

        verifier_positions = np.arange(
            case["proposalPosition"],
            case["proposalPosition"] + expected["verifierTokenCount"],
            dtype=np.int64,
        )
        verifier_rows = self._raw_cache_rows(verifier_positions, case)
        rolled_back = commit_raw_cache_transaction(
            state128, case["proposalPosition"], verifier_rows, 0
        )
        self.assertIs(rolled_back, state128)
        np.testing.assert_array_equal(rolled_back.rows, before)

        committed = commit_raw_cache_transaction(
            state128,
            case["proposalPosition"],
            verifier_rows,
            expected["acceptedRows"],
        )
        self.assertEqual(
            committed.token_start, expected["afterCommitTokenStart"]
        )
        logical_committed = logical_raw_cache(committed)
        logical_samples = [
            item for item in expected["afterCommitSamples"] if "logical" in item
        ]
        physical_samples = [
            item for item in expected["afterCommitSamples"] if "physical" in item
        ]
        self._assert_raw_samples(
            logical_committed, logical_samples, "logical"
        )
        self._assert_raw_samples(
            committed.rows, physical_samples, "physical"
        )
        self.assertFalse(np.shares_memory(committed.rows, state128.rows))
        np.testing.assert_array_equal(state128.rows, before)

        # Validation is transactional: a discontinuity or invalid prefix
        # cannot partially overwrite even the wrap slots.
        with self.assertRaisesRegex(ValueError, "does not follow"):
            commit_raw_cache_transaction(
                state128,
                case["proposalPosition"] + 1,
                verifier_rows,
                expected["acceptedRows"],
            )
        with self.assertRaisesRegex(ValueError, "valid target-row prefix"):
            commit_raw_cache_transaction(
                state128,
                case["proposalPosition"],
                verifier_rows,
                expected["verifierTokenCount"] + 1,
            )
        np.testing.assert_array_equal(state128.rows, before)

    def test_dspark_attention_pins_physical_order_and_bf16_boundaries(
        self,
    ) -> None:
        state, queries, drafts, sinks = self._dspark_attention_inputs()
        result = dspark_attention_official(
            queries, state, drafts, sinks, stage=1
        )
        self.assertEqual(result.output.shape, (5, 64, 512))
        self.assertEqual(result.physical_kv.shape, (133, 512))
        self.assertEqual(
            hashlib.sha256(result.output.astype("<f4").tobytes()).hexdigest(),
            "a326474e55da66818fcc1a66d97d97501209698db8d814b8b70432dc32997de2",
        )
        output_bits = result.output.astype(np.float32).view(np.uint32)
        self.assertTrue(np.all((output_bits & np.uint32(0xFFFF)) == 0))

        # A chronological staging implementation looks plausible but is not
        # pinned-kernel equivalent after wrap.  This fixture is deliberately
        # sensitive to the official physical 0..127 block enumeration.
        chronological_state = type(state)(
            state.capacity,
            state.token_start,
            state.length,
            logical_raw_cache(state),
        )
        chronological = dspark_attention_official(
            queries, chronological_state, drafts, sinks, stage=1
        )
        maximum = float(np.max(np.abs(
            result.output - chronological.output
        )))
        self.assertEqual(maximum, 0.00048828125)
        self.assertGreater(
            int(np.count_nonzero(result.output != chronological.output)),
            10000,
        )
        one_row_state = type(state)(
            state.capacity, 0, 1, np.array(state.rows, copy=True)
        )
        with self.assertRaisesRegex(ValueError, "at least two committed"):
            dspark_attention_official(
                queries, one_row_state, drafts, sinks, stage=1
            )
        with self.assertRaisesRegex(ValueError, r"\[2, 128\]"):
            mlx_optional.dspark_attention_official(
                queries, state.rows[1], 1, drafts[1], sinks
            )

    def test_stage_zero_attention_half_freezes_final_0731_boundaries(
        self,
    ) -> None:
        expected = {
            2: {
                "positions": [2, 3, 4, 5, 6],
                "q_roped":
                    "c4af08ae39bb3541f37de67e55353eba4892a7fdc8ca7ebe22af466c2411bcc7",
                "kv_roped":
                    "92e588e8d190862bba215a26a6c3311ecc281f7d2e7d05b2115c5abee6d6cc9f",
                "kv_stored":
                    "2bf3bb6e70ceffb31053ff6d4d8080c3cbd7487f2e48db919e27ca4c47ece0af",
                "attention_output":
                    "a72d3d257b520857919b0f7c7f74a0d3d96de6219b7a9571a7a54689d8ca2db4",
                "attention_inverse_roped":
                    "f3cc9dbfec4c941f8e2af195c8061b18980d952c4d1e042c8b0e5a0da667e654",
                "output_a":
                    "c6995846d183f18795767f2b2135eef1be5f662ab2f5c6e650ed999b012c6829",
                "output_b":
                    "a85a891f87a400b428279e69eaf12e9947bccda060b1ba17e665f97e51e5e930",
                "hc_post_output":
                    "908b623444a23182afa2e77345fcad5ff6621a54a4d0759e31b81d3d51ea9670",
            },
            128: {
                "positions": [130, 131, 132, 133, 134],
                "q_roped":
                    "4c66d4e2459b826612dcdd5d6e40cb5ee6eb1a95191c5f7229a1c19feb0d76d6",
                "kv_roped":
                    "779baae533c21b3ef9f2c9fe6c936ea1365a3b71d391d6ab203ad96d393b0f14",
                "kv_stored":
                    "ec4457c2b6b07dfa71bb4d4a34340b1f4c9396ab73c7158dd9872d5f5f5ea47a",
                "attention_output":
                    "2657472630bb63bcc52d53833f4326cd5b4ee56cfa1c3382a96f8b192ec3be23",
                "attention_inverse_roped":
                    "185399963ec57e0ce27a006e7f44c7865de3b5b16c6039e9a5eb45d6f626c4f9",
                "output_a":
                    "999c69eaff63752dc81822a734ef20574acc1bee0c4a22085c554807ddabd6e0",
                "output_b":
                    "cb595fc41764a08e5259630e88b5ac6e8af1a599ef003c4aa06f51801d5ec6eb",
                "hc_post_output":
                    "ac303a4a96c2fd9d03042c55cefc6f2f62e229ad794535f177f98d681e99e8bd",
            },
        }
        common_digests = {
            "hidden_input":
                "04795e84f99fd90ce64f6848fd7b6bd4f9e0cc91c320acb131931a688ce0e121",
            "hc_pre_output":
                "bee72ec3bd81cfae3bd13980ca6cdd9201fe9e350d406540e4c69d56475a2a9b",
            "attention_normalized":
                "89d1ce6729a9a49b72df6d724044365c1c4337f218a280d2f0cb9869964fc010",
            "q_a":
                "676f6cdb2d3a5af51fd7e4c9f643328d5646e99af88a0e9e8858fc4cafd662ad",
            "q_a_normalized":
                "e2edae32ec96e877d425d2e3045ba8f4518f1cfba3aee860179a0f3dfe3351fd",
            "q_b":
                "b3e26819e09ad64681813691990d0c4915d7c33010490e1516c9746eafee0202",
            "q_head_normalized":
                "9eef1d0a375260173e947940fc3294ba03c67d58eb95f8cedeb4e80af4aca22a",
            "kv_projected":
                "1555036a1c4e3167b86d1a74eb0e4b0da8abfe63e1578809127ff8db111a5059",
            "kv_normalized":
                "14a362457d1b631903d9a56be9f5d9c1fe15de8284d14faa87a72889225a79b9",
            "kv_nonrope_scales":
                "57c0e9e040e2ecb200d5b6695df6c53edb9c166cc4552aa32614fbeaeadb6b46",
        }
        boundary_shapes = {
            "hidden_input": (5, 4, 8),
            "hc_pre_output": (5, 8),
            "attention_normalized": (5, 8),
            "q_a": (5, 8),
            "q_a_normalized": (5, 8),
            "q_b": (5, 64, 512),
            "q_head_normalized": (5, 64, 512),
            "q_roped": (5, 64, 512),
            "kv_projected": (5, 512),
            "kv_normalized": (5, 512),
            "kv_roped": (5, 512),
            "kv_stored": (5, 512),
            "attention_output": (5, 64, 512),
            "attention_inverse_roped": (5, 64, 512),
            "output_a": (5, 8, 2),
            "output_b": (5, 8),
            "hc_post_output": (5, 4, 8),
        }

        full_inputs: dict[str, object] | None = None
        full_result: object | None = None
        for count in (2, 128):
            inputs = self._stage_zero_attention_half_inputs(count)
            q_b_blocks = np.asarray(
                inputs["q_b_weight"], dtype=np.float32
            ).reshape(64, 512, 8)
            self.assertEqual(
                len({
                    hashlib.sha256(block.tobytes()).digest()
                    for block in q_b_blocks
                }),
                64,
            )
            state = inputs["raw_cache"]
            before = np.array(state.rows, copy=True)
            result = stage_zero_attention_half(**inputs)
            self.assertTrue(np.all(np.any(
                result.q_a_normalized != 0.0, axis=0
            )))
            for field in ("q_b", "q_roped"):
                head_blocks = getattr(result, field).transpose(1, 0, 2)
                self.assertEqual(
                    len({block.tobytes() for block in head_blocks}), 64,
                    f"{field} must retain 64 live head signatures",
                )
            self.assertTrue(np.all(np.any(
                result.output_a != 0.0, axis=0
            )))
            self.assertEqual(
                int(np.count_nonzero(result.output_a)), result.output_a.size
            )
            self.assertEqual(
                result.absolute_positions.tolist(), expected[count]["positions"]
            )
            self.assertEqual(state.length, count)
            self.assertEqual(state.token_start, 0 if count == 2 else 2)
            np.testing.assert_array_equal(state.rows, before)
            self.assertEqual(result.kv_nonrope_scales.shape, (5, 7))
            for field, shape in boundary_shapes.items():
                boundary = getattr(result, field)
                self.assertEqual(boundary.shape, shape, field)
                self._assert_bfloat16_boundary(boundary)
            for field, digest in common_digests.items():
                self.assertEqual(
                    self._array_digest(getattr(result, field), "<f4"),
                    digest,
                    field,
                )
            idempotent_inputs = dict(inputs)
            idempotent_inputs["hidden_input"] = result.hidden_input
            idempotent = stage_zero_attention_half(**idempotent_inputs)
            for field in result.__dataclass_fields__:
                np.testing.assert_array_equal(
                    getattr(idempotent, field), getattr(result, field),
                    err_msg=f"already-BF16 input changed {field}",
                )
            for field, digest in expected[count].items():
                if field == "positions":
                    continue
                self.assertEqual(
                    self._array_digest(getattr(result, field), "<f4"),
                    digest,
                    f"C={count} {field}",
                )
            if count == 128:
                for field, unique_count in (
                    ("kv_projected", 20),
                    ("kv_normalized", 61),
                    ("kv_roped", 264),
                    ("kv_stored", 236),
                ):
                    boundary = getattr(result, field)
                    self.assertEqual(
                        int(np.count_nonzero(boundary)), boundary.size, field
                    )
                    self.assertEqual(
                        int(np.unique(boundary).size), unique_count, field
                    )
                fp8_changed = (
                    result.kv_roped[:, :-64] != result.kv_stored[:, :-64]
                ).reshape(5, 7, 64)
                self.assertEqual(int(np.count_nonzero(fp8_changed)), 2178)
                np.testing.assert_array_equal(
                    np.count_nonzero(fp8_changed, axis=(0, 2)),
                    np.asarray([310, 312, 310, 313, 311, 311, 311]),
                )
                full_inputs, full_result = inputs, result

        assert full_inputs is not None and full_result is not None
        full_state = full_inputs["raw_cache"]
        chronological_state = type(full_state)(
            full_state.capacity,
            full_state.token_start,
            full_state.length,
            logical_raw_cache(full_state),
        )
        chronological_inputs = dict(full_inputs)
        chronological_inputs["raw_cache"] = chronological_state
        chronological = stage_zero_attention_half(**chronological_inputs)
        self.assertEqual(
            int(np.count_nonzero(
                full_result.attention_output != chronological.attention_output
            )),
            83,
        )
        self.assertEqual(
            float(np.max(np.abs(
                full_result.attention_output
                - chronological.attention_output
            ))),
            0.000244140625,
        )

    def test_stage_zero_attention_half_rejects_skipped_bf16_boundaries(
        self,
    ) -> None:
        inputs = self._stage_zero_attention_half_inputs(128)
        result = stage_zero_attention_half(**inputs)

        # Negative control 1: the stage input itself is published BF16 before
        # HC-pre and is also the only legal HC-post residual.
        raw_hidden = np.asarray(inputs["hidden_input"], dtype=np.float32)
        self.assertEqual(
            int(np.count_nonzero(raw_hidden != result.hidden_input)), 160
        )
        skipped_input_hc, skipped_input_split = hc_pre(
            raw_hidden,
            inputs["hc_function_weight"],
            inputs["hc_scale"],
            inputs["hc_base"],
        )
        skipped_input_hc = self._round_fixture_bfloat16(skipped_input_hc)
        self.assertEqual(
            int(np.count_nonzero(
                skipped_input_hc != result.hc_pre_output
            )),
            14,
        )
        skipped_input_residual = self._round_fixture_bfloat16(hc_post(
            result.output_b, raw_hidden, skipped_input_split
        ))
        self.assertEqual(
            int(np.count_nonzero(
                skipped_input_residual != result.hc_post_output
            )),
            88,
        )
        original_round = dspark_reference._round_bfloat16
        round_calls = 0

        def skip_first_input_round(value: np.ndarray) -> np.ndarray:
            nonlocal round_calls
            round_calls += 1
            if round_calls == 1:
                return np.asarray(value, dtype=np.float32)
            return original_round(value)

        with mock.patch.object(
            dspark_reference,
            "_round_bfloat16",
            side_effect=skip_first_input_round,
        ):
            skipped_input_end_to_end = stage_zero_attention_half(**inputs)
        self.assertEqual(round_calls, 26)
        self.assertEqual(
            int(np.count_nonzero(
                skipped_input_end_to_end.attention_output
                != result.attention_output
            )),
            640,
        )
        self.assertEqual(
            int(np.count_nonzero(
                skipped_input_end_to_end.hc_post_output
                != result.hc_post_output
            )),
            92,
        )

        # Negative control 2: HC-pre is specified to return to the model dtype
        # before attn_norm.  Keeping its float32 reduction live changes 15
        # frozen BF16 lanes at the next publication.
        unrounded_hc, _ = hc_pre(
            result.hidden_input,
            inputs["hc_function_weight"],
            inputs["hc_scale"],
            inputs["hc_base"],
        )
        unrounded_hc = np.asarray(unrounded_hc, dtype=np.float32)
        variance = np.mean(
            np.square(unrounded_hc), axis=-1, keepdims=True,
            dtype=np.float32,
        )
        skipped_pre_attn_norm = self._round_fixture_bfloat16(
            unrounded_hc
            * (np.float32(1.0) / np.sqrt(variance + np.float32(1.0e-6)))
            * np.asarray(inputs["attention_norm_weight"], dtype=np.float32)
        )
        self.assertEqual(
            int(np.count_nonzero(
                skipped_pre_attn_norm != result.attention_normalized
            )),
            15,
        )
        self.assertNotEqual(
            self._array_digest(skipped_pre_attn_norm, "<f4"),
            self._array_digest(result.attention_normalized, "<f4"),
        )

        # Negative control 3: per-head Q norm is not float32 RMSNorm.  The
        # pinned BF16 expression publishes square, mean, rsqrt and multiply.
        q_b = result.q_b.astype(np.float32)
        wrong_q_variance = np.mean(
            np.square(q_b), axis=-1, keepdims=True, dtype=np.float32
        )
        wrong_q_norm = self._round_fixture_bfloat16(
            q_b * (
                np.float32(1.0)
                / np.sqrt(wrong_q_variance + np.float32(1.0e-6))
            )
        )
        self.assertEqual(
            int(np.count_nonzero(
                wrong_q_norm != result.q_head_normalized
            )),
            52096,
        )

        # Negative control 4: wo_a publishes BF16 before wo_b.  Feeding the
        # unrounded grouped projection directly into wo_b changes 19
        # final BF16 lanes for the wrapped fixture.
        grouped = result.attention_inverse_roped.astype(np.float32).reshape(
            5, 8, 8 * 512
        )
        unrounded_output_a = np.einsum(
            "qgd,grd->qgr",
            grouped,
            np.asarray(inputs["output_a_weight"], dtype=np.float32),
            dtype=np.float32,
            optimize=False,
        )
        skipped_output_a_store = self._round_fixture_bfloat16(
            unrounded_output_a.reshape(5, -1)
            @ np.asarray(inputs["output_b_weight"], dtype=np.float32).T
        )
        self.assertEqual(
            int(np.count_nonzero(
                skipped_output_a_store != result.output_b
            )),
            19,
        )
        self.assertNotEqual(
            self._array_digest(skipped_output_a_store, "<f4"),
            self._array_digest(result.output_b, "<f4"),
        )

        # Negative control 5: official flattening is group-major then rank.
        # With rank >= 2, a plausible rank-major transpose is observable.
        wrong_output_layout = self._round_fixture_bfloat16(
            np.transpose(result.output_a, (0, 2, 1)).reshape(5, -1)
            @ np.asarray(inputs["output_b_weight"], dtype=np.float32).T
        )
        self.assertEqual(
            int(np.count_nonzero(wrong_output_layout != result.output_b)),
            40,
        )

        # Negative control 6: Q-B rows are head-specific.  Rotating the 64
        # head blocks must not preserve sparse attention or its digest.
        permuted_inputs = dict(inputs)
        q_b_weight = np.asarray(inputs["q_b_weight"], dtype=np.float32)
        permuted_inputs["q_b_weight"] = np.roll(
            q_b_weight.reshape(64, 512, 8), shift=1, axis=0
        ).reshape(64 * 512, 8)
        permuted = stage_zero_attention_half(**permuted_inputs)
        for field in ("q_b", "q_roped", "attention_output"):
            changed_heads = (
                getattr(permuted, field) != getattr(result, field)
            ).reshape(5, 64, -1).any(axis=(0, 2))
            self.assertTrue(
                np.all(changed_heads),
                f"Q-B head rotation left a fixed point in {field}",
            )
        self.assertEqual(
            int(np.count_nonzero(
                permuted.attention_output != result.attention_output
            )),
            1429,
        )

    def test_physical_stage_zero_payloads_match_q8_f16_contract(self) -> None:
        # Frozen directly from gguf-tools/quants.c.  The first block proves C
        # roundf ties away from zero; the second proves that codes use the
        # original F32 d while dequantization reopens the stored F16 d.
        golden_source = np.zeros((2, 32), dtype=np.float32)
        golden_source[0, :6] = [127.0, -127.0, 0.5, -0.5, 1.5, -1.5]
        golden_source[1, :6] = [
            1.0, -1.0, 0.5, -0.5,
            np.float32(1.0 / 127.0), np.float32(-1.0 / 127.0),
        ]
        scale_discriminator = np.asarray(
            [0x3B810001], dtype=np.uint32
        ).view(np.float32)[0]
        golden_source[1, 6:8] = [
            scale_discriminator, -scale_discriminator
        ]
        golden_payload = bytes.fromhex(
            "003c7f8101ff02fe0000000000000000000000000000000000000000000000000000"
            "08207f8140c001ff0000000000000000000000000000000000000000000000000000"
        )
        packed_golden = pack_q8_0(golden_source)
        self.assertEqual(packed_golden.payload, golden_payload)
        self.assertEqual(
            hashlib.sha256(golden_payload).hexdigest(),
            "442988c83babde0175668c56b6a864bca70a8a51bbc0ae9b1ba4f4b33ccaa994",
        )
        np.testing.assert_array_equal(
            packed_golden.dequantized[0, :6],
            np.asarray([127, -127, 1, -1, 2, -2], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            packed_golden.dequantized[1, :6],
            np.asarray([
                0.99993896484375, -0.99993896484375,
                0.50390625, -0.50390625,
                0.00787353515625, -0.00787353515625,
            ], dtype=np.float32),
        )
        original_d = np.float32(1.0) / np.float32(127.0)
        reopened_d = np.frombuffer(
            golden_payload[34:36], dtype="<f2"
        ).astype(np.float32)[0]
        original_scaled = np.float32(
            scale_discriminator * np.float32(1.0 / original_d)
        )
        reopened_scaled = np.float32(
            scale_discriminator * np.float32(1.0 / reopened_d)
        )
        self.assertLess(original_scaled, np.float32(0.5))
        self.assertGreater(reopened_scaled, np.float32(0.5))
        def round_away(value: np.float32) -> int:
            return int(np.copysign(
                np.floor(np.abs(value) + np.float32(0.5)), value
            ))

        self.assertEqual(
            [round_away(original_scaled), round_away(-original_scaled)],
            [0, 0],
        )
        self.assertEqual(
            [round_away(reopened_scaled), round_away(-reopened_scaled)],
            [1, -1],
        )
        self.assertEqual(golden_payload[34 + 2 + 6:34 + 2 + 8], b"\0\0")

        expected_payloads = {
            "hc_function_weight": (0, 6144,
                "4ccb8dccef22e4ea75135b33950e1c1d0b27962b83c2149c92ca059622054b99"),
            "hc_scale": (6144, 12,
                "59727b504e27a54d55b2183e4d218ffc4388e9ad46bc149b3fb59b5228349fc2"),
            "hc_base": (6208, 96,
                "6810f80a9ad75534de871ddf2477436f611481a1433311f2eceb5a9401e15511"),
            "attention_norm_weight": (6336, 128,
                "d5aa67b0772c9b552f4fbb972085cae582b2c0b47e8f1e38fbadac8638e83771"),
            "q_a_weight": (6464, 1088,
                "425cea3fa14fe7a5a38d1456908c9152c1d29a67899e9dc34a73d8d5c0d9e688"),
            "q_a_norm_weight": (7552, 128,
                "b638277a8690e175a9137feff1e43c067f9faf4e2f600caf468fb05b0403b717"),
            "q_b_weight": (7680, 1114112,
                "383b935ecc401de51e2dbb90c847276664f1bf2dd52642c43217aa83c0e138ed"),
            "kv_weight": (1121792, 17408,
                "fedfb98e161898f5d182a7f6d4405b4e80eb5721aede2c1352ed14d651819f0c"),
            "kv_norm_weight": (1139200, 2048,
                "1a397dca862f8f5b132e0a5a72de5e81e26ce975c92f95c5cf6498ef733882f0"),
            "attention_sinks": (1141248, 256,
                "c50a37025f0389dc726442d3f2876da4f1e83a6e413dc1985ca28237d8e1bebc"),
            "output_a_weight": (1141504, 139264,
                "640796379d715fd577dd23b5cbd4367e4ef5f6c48e3e7b18b039ac87241bda79"),
            "output_b_weight": (1280768, 1088,
                "da9e31cab8b4fd9bc7e8b2b91f9fa7219c8668eac54ba64739511b39d4fb837d"),
        }
        fixture = build_physical_stage_zero_fixture(128)
        manifest = payload_manifest(fixture)
        self.assertEqual(manifest["fixture_version"], 1)
        self.assertEqual(manifest["geometry"], {
            "proposal_rows": 5,
            "hc_lanes": 4,
            "hidden_width": 32,
            "q_rank": 32,
            "attention_heads": 64,
            "head_width": 512,
            "output_groups": 8,
            "output_rank": 4,
        })
        self.assertEqual(manifest["alignment"], PHYSICAL_MODEL_ALIGNMENT)
        self.assertEqual(manifest["blob_bytes"], 1281856)
        self.assertEqual(
            manifest["blob_sha256"],
            "1388a4a205ae61c59a25df4a03af312e2dea1fb13d35f6503362f06dd0ee1492",
        )
        self.assertEqual(manifest["raw_hidden_input"], {
            "shape": [5, 4, 32],
            "bytes": 5 * 4 * 32 * 4,
            "sha256":
                "e15d38302793fb96779672dcc38a99ef59b0de2f07ea25c17c348d37335dad57",
        })
        self.assertEqual(manifest["stage_zero_raw_cache"], {
            "capacity": 128,
            "token_start": 2,
            "length": 128,
            "shape": [128, 512],
            "bytes": 128 * 512 * 4,
            "sha256":
                "d085299feb54f6010b64b7a7550dbb3b90d4c03c800de9384db0aa2fa36ea338",
        })
        partial_manifest = payload_manifest(
            build_physical_stage_zero_fixture(2)
        )
        self.assertEqual(partial_manifest["stage_zero_raw_cache"], {
            "capacity": 128,
            "token_start": 0,
            "length": 2,
            "shape": [128, 512],
            "bytes": 128 * 512 * 4,
            "sha256":
                "e27748d96d6d36cd5b12f42a710eb76d0e29b27c774fc11d153e9536d4526c9d",
        })
        expected_required_paths = {
            "q_a_weight": "generic_mm_half_staged",
            "q_b_weight": "generic_mm_half_staged",
            "kv_weight": "generic_mm_half_staged",
            "output_a_weight": "direct_grouped_q8_matvec_f32",
            "output_b_weight": "generic_mm_half_staged",
        }
        self.assertEqual(
            manifest["required_q8_paths"], expected_required_paths
        )
        previous_end = 0
        for name, (offset, size, digest) in expected_payloads.items():
            weight = fixture.packed_weights[name]
            self.assertEqual(fixture.model_offsets[name], offset)
            self.assertEqual(len(weight.payload), size)
            self.assertEqual(weight.sha256, digest)
            self.assertEqual(offset % PHYSICAL_MODEL_ALIGNMENT, 0)
            self.assertEqual(
                fixture.model_blob[offset:offset + size], weight.payload
            )
            self.assertEqual(
                fixture.model_blob[previous_end:offset],
                bytes(offset - previous_end),
            )
            previous_end = offset + size
            np.testing.assert_array_equal(
                fixture.inputs[name], weight.dequantized
            )
            if weight.storage == "Q8_0":
                self.assertGreater(int(np.count_nonzero(
                    fixture.ideal_weights[name] != weight.dequantized
                )), 0)
            else:
                np.testing.assert_array_equal(
                    fixture.ideal_weights[name], weight.dequantized
                )
        self.assertEqual(previous_end, len(fixture.model_blob))

        for name in expected_required_paths:
            weight = fixture.packed_weights[name]
            input_width = weight.logical_shape[-1]
            output_rows = int(np.prod(weight.logical_shape[:-1]))
            self.assertEqual(manifest["weights"][name]["oracle_shape"],
                             list(weight.logical_shape))
            self.assertEqual(manifest["weights"][name]["gguf_ne"],
                             [input_width, output_rows])
            self.assertEqual(manifest["weights"][name]["row_bytes"],
                             input_width // 32 * 34)
            self.assertEqual(manifest["weights"][name]["output_rows"],
                             output_rows)
            payload_blocks = np.frombuffer(
                weight.payload, dtype=np.uint8
            ).reshape(-1, 34)
            scales = np.ascontiguousarray(
                payload_blocks[:, :2]
            ).view("<f2").astype(np.float32).reshape(-1)
            self.assertTrue(np.all(
                (scales == 0.0) | (scales == np.float32(1.0 / 256.0))
            ))
            rows = int(np.prod(weight.logical_shape[:-1]))
            codes = payload_blocks[:, 2:].view(np.int8).reshape(rows, -1)
            np.testing.assert_array_equal(
                np.count_nonzero(codes, axis=1), np.ones(rows, dtype=np.int64)
            )
            self.assertTrue(np.all(
                np.asarray(weight.dequantized, dtype=np.float32)
                == np.asarray(weight.dequantized, dtype=np.float16).astype(np.float32)
            ))

        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            pack_q8_0(np.zeros((2, 31), dtype=np.float32))
        invalid = np.zeros((1, 32), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            pack_q8_0(invalid)
        with self.assertRaisesRegex(ValueError, "size"):
            unpack_q8_0(bytes(33), (1, 32))
        with self.assertRaisesRegex(ValueError, "only C=2 or C=128"):
            build_physical_stage_zero_fixture(True)

    def test_physical_payloads_match_current_c_quantizer(self) -> None:
        """Cross-check Python bytes against the repository C implementation."""

        with tempfile.TemporaryDirectory(
            prefix="ds4q-cross-language-"
        ) as temporary:
            if sys.platform == "darwin":
                library_path = Path(temporary) / "libds4q.dylib"
                shared_flags = ["-dynamiclib"]
            else:
                library_path = Path(temporary) / "libds4q.so"
                shared_flags = ["-shared", "-fPIC"]
            compile_command = [
                os.environ.get("CC", "cc"),
                *shared_flags,
                "-O2",
                "-Wall",
                "-Wextra",
                "-std=c11",
                "-I",
                str(ROOT / "gguf-tools"),
                "-o",
                str(library_path),
                str(ROOT / "gguf-tools" / "quants.c"),
                "-lm",
                "-pthread",
            ]
            compiled = subprocess.run(
                compile_command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if compiled.returncode:
                self.fail(
                    "current gguf-tools/quants.c did not compile for the "
                    "cross-language oracle\n"
                    f"stdout:\n{compiled.stdout}\n"
                    f"stderr:\n{compiled.stderr}"
                )

            library = ctypes.CDLL(str(library_path))
            quantize = library.ds4q_quantize_chunk
            quantize.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
            ]
            quantize.restype = ctypes.c_size_t
            f32_to_f16 = library.ds4q_f32_to_f16_row
            f32_to_f16.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.c_int64,
            ]
            f32_to_f16.restype = None

            def c_pack_q8_0(value: np.ndarray) -> bytes:
                source = np.ascontiguousarray(value, dtype=np.float32)
                width = source.shape[-1]
                rows = int(np.prod(source.shape[:-1], dtype=np.int64))
                expected_bytes = rows * (width // 32) * 34
                destination = (ctypes.c_uint8 * expected_bytes)()
                written = quantize(
                    8,
                    source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(destination, ctypes.c_void_p),
                    0,
                    rows,
                    width,
                    None,
                )
                self.assertEqual(written, expected_bytes)
                return bytes(destination)

            golden = np.zeros((2, 32), dtype=np.float32)
            golden[0, :6] = [127.0, -127.0, 0.5, -0.5, 1.5, -1.5]
            golden[1, :6] = [
                1.0, -1.0, 0.5, -0.5,
                np.float32(1.0 / 127.0),
                np.float32(-1.0 / 127.0),
            ]
            discriminator = np.asarray(
                [0x3B810001], dtype=np.uint32
            ).view(np.float32)[0]
            golden[1, 6:8] = [discriminator, -discriminator]

            sparse_4096 = np.zeros((3, 4096), dtype=np.float32)
            sparse_4096[0, 0] = np.float32(127.0 / 256.0)
            sparse_4096[1, 4095] = np.float32(-127.0 / 256.0)
            sparse_4096[2, 2017] = np.float32(127.0 / 256.0)

            generator = np.random.default_rng(0xD54A)
            dense_codes = generator.integers(
                -126, 127, size=(11, 96), dtype=np.int16
            )
            dense_codes.reshape(-1, 32)[:, 0] = np.where(
                (np.arange(dense_codes.size // 32) & 1) == 0, 127, -127
            )
            seeded_dense = (
                dense_codes.astype(np.float32) * np.float32(1.0 / 256.0)
            )

            fixture = build_physical_stage_zero_fixture(128)
            q8_cases = {
                "golden_scale_discriminator": golden,
                "fixture32": fixture.ideal_weights["q_a_weight"],
                "sparse4096_zero_blocks": sparse_4096,
                "seeded_dense": seeded_dense,
            }
            q8_cases.update({
                f"fixture_{name}": fixture.ideal_weights[name]
                for name in (
                    "q_b_weight",
                    "kv_weight",
                    "output_a_weight",
                    "output_b_weight",
                )
            })
            for label, source in q8_cases.items():
                with self.subTest(payload=label):
                    c_payload = c_pack_q8_0(source)
                    self.assertEqual(c_payload, pack_q8_0(source).payload)
                    if label.startswith("fixture_") or label == "fixture32":
                        weight_name = (
                            "q_a_weight" if label == "fixture32"
                            else label.removeprefix("fixture_")
                        )
                        self.assertEqual(
                            c_payload,
                            fixture.packed_weights[weight_name].payload,
                        )

            hc_source = np.ascontiguousarray(
                fixture.ideal_weights["hc_function_weight"],
                dtype=np.float32,
            ).reshape(-1)
            c_f16 = (ctypes.c_uint16 * hc_source.size)()
            f32_to_f16(
                hc_source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_f16,
                hc_source.size,
            )
            self.assertEqual(
                bytes(c_f16),
                fixture.packed_weights["hc_function_weight"].payload,
            )

    def test_physical_stage_zero_freezes_payload_derived_boundaries(self) -> None:
        common = {
            "hidden_input": "84b3088e8ba5e270e498bb24ace221c1790a6bedd88eb65351143894366eb870",
            "hc_pre_output": "ff7b9064c8d35fbc6410e40cac370530dfe492b52e6ffde5c9711d1810092ba3",
            "attention_normalized": "79395379438bf9bdfbf7d56687c193f78ca8db594b3f49a93a496a9de95d3022",
            "q_a": "31ae0c918831c05069b5d78191269416fd97152d2bcfb4ccdbeb12f5c35f2764",
            "q_a_normalized": "e68b314d4beada722e99ab0fc07451cde2c44ff5c28779b8ad82928b8524e37d",
            "q_b": "a21ff121b34184f686f9b1e987bec618f32403ffdafb09514be8a3c104e9c8ba",
            "q_head_normalized": "3a16369a4b7563951550e33128a19482b558cdd39d8ec0e1b206ef5fd93be1de",
            "kv_projected": "748c0b94b250ff8569e9ae06c0ce24de624f301d5dbf13c163ef13aaf80259a1",
            "kv_normalized": "fd61dd84c3db71d59d4aeaf85f3f74bf3b74544c7d743d421a28ce747fb87c3a",
            "kv_nonrope_scales": "57c0e9e040e2ecb200d5b6695df6c53edb9c166cc4552aa32614fbeaeadb6b46",
        }
        specific = {
            2: {
                "positions": [2, 3, 4, 5, 6],
                "q_roped": "d33ef0cb710957fd468e7a354fb9e5a3ffe950a14ed4c66bffda3a31d77e4a2e",
                "kv_roped": "6444b8649c5a5aa491045e1ba5ee76aecdf2b15b14174f96dcab3653fdfd93dc",
                "kv_stored": "3905eb6d3b0ee06cbdab1f52914f425483e1549cbfebcb68b33c2c8e1d42c03a",
                "attention_output": "4e040df2c259d4f07383a32661e76e7e751d4fb2f2cf6dc18a6a297af1a86484",
                "attention_inverse_roped": "4a1f8f6e9a601116517cd9a47cf9428eb4c755c378c00df38c27bf3bef33b8be",
                "output_a": "0abb6633e98af7ddccbb1b1468e469e7c79febb9e954457bc0643f5cb712d130",
                "output_b": "5596c4de9d19abb8802a1315682c231c03f8db1368e888f17884d5e80239f35f",
                "hc_post_output": "6405889c5b0a836a19700880ea80e1d617deb0b83e784dbf937c015baf289135",
            },
            128: {
                "positions": [130, 131, 132, 133, 134],
                "q_roped": "412662520f68811c3dce9da664d3898df83cf05630efdaabba17eb7f8d321a12",
                "kv_roped": "2acd0b7f04d9f0ada1c9f4604c7349fe0e2b7a0a513de770915e8def02035fe0",
                "kv_stored": "754aef76ba986691af72a92819b9563fa191aa7501af41dccc9928e1167b6380",
                "attention_output": "c2d1b7b01469a2cf540af8de106509213017fe5157784253db6700bec5c45b13",
                "attention_inverse_roped": "a816c89adbfb97e27cc844e00758a454ab6498d2199cff3e8188f66c4e008014",
                "output_a": "9ee20059b62050e73231bbe74396983e6483337533f37a21bed7744f50fe425e",
                "output_b": "2d169b04ad0e5c20670e5a39daf149f93fbdb3684a3eea772fd8cf26cd7d352b",
                "hc_post_output": "f73b649e92b56e39f3ffe71710800793afff8fb7433a849851613d4555180b9f",
            },
        }
        expected_shapes = {
            "hidden_input": (5, 4, PHYSICAL_HIDDEN_WIDTH),
            "hc_pre_output": (5, PHYSICAL_HIDDEN_WIDTH),
            "attention_normalized": (5, PHYSICAL_HIDDEN_WIDTH),
            "q_a": (5, PHYSICAL_Q_RANK),
            "q_a_normalized": (5, PHYSICAL_Q_RANK),
            "q_b": (5, 64, 512),
            "q_head_normalized": (5, 64, 512),
            "q_roped": (5, 64, 512),
            "kv_projected": (5, 512),
            "kv_normalized": (5, 512),
            "kv_roped": (5, 512),
            "kv_stored": (5, 512),
            "attention_output": (5, 64, 512),
            "attention_inverse_roped": (5, 64, 512),
            "output_a": (5, 8, PHYSICAL_OUTPUT_RANK),
            "output_b": (5, PHYSICAL_HIDDEN_WIDTH),
            "hc_post_output": (5, 4, PHYSICAL_HIDDEN_WIDTH),
        }

        for count in (2, 128):
            fixture = build_physical_stage_zero_fixture(count)
            state = fixture.inputs["raw_cache"]
            before = np.array(state.rows, copy=True)
            result = stage_zero_attention_half(**fixture.inputs)
            np.testing.assert_array_equal(state.rows, before)
            raw_hidden = np.asarray(
                fixture.inputs["hidden_input"], dtype=np.float32
            )
            self.assertEqual(
                int(np.count_nonzero(raw_hidden != result.hidden_input)), 640
            )
            np.testing.assert_array_equal(
                self._round_fixture_bfloat16(raw_hidden), result.hidden_input
            )
            # Stage zero replaces draft_rows[0] with payload-derived KV.  The
            # stage-1/2 template is validated for shape/finite input only and
            # is intentionally irrelevant to this stage-zero attention result.
            alternate_drafts = dict(fixture.inputs)
            alternate = np.array(
                alternate_drafts["other_stage_draft_rows"], copy=True
            )
            alternate[1:] *= np.float32(-3.0)
            alternate_drafts["other_stage_draft_rows"] = alternate
            alternate_result = stage_zero_attention_half(**alternate_drafts)
            for field in result.__dataclass_fields__:
                np.testing.assert_array_equal(
                    getattr(alternate_result, field), getattr(result, field),
                    err_msg=f"stage-1/2 draft template changed {field}",
                )
            self.assertEqual(
                result.absolute_positions.tolist(), specific[count]["positions"]
            )
            for field, shape in expected_shapes.items():
                boundary = getattr(result, field)
                self.assertEqual(boundary.shape, shape, field)
                self._assert_bfloat16_boundary(boundary)
            for field, digest in common.items():
                self.assertEqual(
                    self._array_digest(getattr(result, field), "<f4"),
                    digest,
                    field,
                )
            for field, digest in specific[count].items():
                if field == "positions":
                    continue
                self.assertEqual(
                    self._array_digest(getattr(result, field), "<f4"),
                    digest,
                    field,
                )

            self.assertGreaterEqual(np.unique(result.q_a).size, 29)
            self.assertGreaterEqual(np.unique(result.q_b).size, 52)
            self.assertTrue(np.all(np.any(result.q_a != 0.0, axis=0)))
            self.assertEqual(
                len({row.tobytes() for row in result.q_b.transpose(1, 0, 2)}),
                64,
            )
            self.assertEqual(int(np.count_nonzero(result.output_a)), 160)
            self.assertEqual(int(np.count_nonzero(result.output_b)), 160)

            mm_inputs = {
                "q_a_weight": result.attention_normalized,
                "q_b_weight": result.q_a_normalized,
                "kv_weight": result.attention_normalized,
                "output_a_weight": result.attention_inverse_roped,
                "output_b_weight": result.output_a,
            }
            for weight_name, activation in mm_inputs.items():
                np.testing.assert_array_equal(
                    activation,
                    np.asarray(activation, dtype=np.float16).astype(np.float32),
                    err_msg=f"{weight_name} input is not F16-exact",
                )

            squared = self._round_fixture_bfloat16(np.square(result.q_b))
            tree = squared[..., :256] + squared[..., 256:]
            for stride in (128, 64, 32, 16, 8, 4, 2, 1):
                tree[..., :stride] = (
                    tree[..., :stride] + tree[..., stride:2 * stride]
                )
            tree_mean = self._round_fixture_bfloat16(
                tree[..., :1] / np.float32(512.0)
            )
            numpy_mean = self._round_fixture_bfloat16(np.mean(
                squared, axis=-1, keepdims=True, dtype=np.float32
            ))
            np.testing.assert_array_equal(tree_mean, numpy_mean)

    def test_physical_stage_zero_mutations_are_observable(self) -> None:
        fixture = build_physical_stage_zero_fixture(128)
        inputs = dict(fixture.inputs)
        result = stage_zero_attention_half(**inputs)

        original_round = dspark_reference._round_bfloat16
        round_calls = 0

        def skip_ingress_publication(value: np.ndarray) -> np.ndarray:
            nonlocal round_calls
            round_calls += 1
            if round_calls == 1:
                return np.asarray(value, dtype=np.float32)
            return original_round(value)

        with mock.patch.object(
            dspark_reference,
            "_round_bfloat16",
            side_effect=skip_ingress_publication,
        ):
            skipped_ingress = stage_zero_attention_half(**inputs)
        self.assertEqual(round_calls, 26)
        self.assertEqual(int(np.count_nonzero(
            skipped_ingress.hidden_input != result.hidden_input
        )), 640)
        self.assertEqual(int(np.count_nonzero(
            skipped_ingress.hc_post_output != result.hc_post_output
        )), 4)

        state = inputs["raw_cache"]
        chronological_state = type(state)(
            state.capacity, state.token_start, state.length,
            logical_raw_cache(state),
        )
        chronological_inputs = dict(inputs)
        chronological_inputs["raw_cache"] = chronological_state
        chronological = stage_zero_attention_half(**chronological_inputs)
        self.assertEqual(int(np.count_nonzero(
            chronological.attention_output != result.attention_output
        )), 436)
        self.assertEqual(float(np.max(np.abs(
            chronological.attention_output - result.attention_output
        ))), 0.0001220703125)

        rank_inputs = dict(inputs)
        rank_inputs["q_a_weight"] = np.roll(
            np.asarray(inputs["q_a_weight"], dtype=np.float32), 1, axis=0
        )
        rank_mutation = stage_zero_attention_half(**rank_inputs)
        self.assertEqual(int(np.count_nonzero(
            rank_mutation.q_a != result.q_a
        )), 160)
        self.assertEqual(int(np.count_nonzero(
            rank_mutation.attention_output != result.attention_output
        )), 50134)

        head_inputs = dict(inputs)
        q_b_weight = np.asarray(inputs["q_b_weight"], dtype=np.float32)
        head_inputs["q_b_weight"] = np.roll(
            q_b_weight.reshape(64, 512, 32), 1, axis=0
        ).reshape(64 * 512, 32)
        head_mutation = stage_zero_attention_half(**head_inputs)
        changed_heads = (
            head_mutation.attention_output != result.attention_output
        ).reshape(5, 64, 512).any(axis=(0, 2))
        self.assertTrue(np.all(changed_heads))
        self.assertEqual(int(np.count_nonzero(
            head_mutation.attention_output != result.attention_output
        )), 49198)

        wrong_layout = self._round_fixture_bfloat16(
            np.transpose(result.output_a, (0, 2, 1)).reshape(5, -1)
            @ np.asarray(inputs["output_b_weight"], dtype=np.float32).T
        )
        self.assertEqual(
            int(np.count_nonzero(wrong_layout != result.output_b)), 150
        )

        q_a_payload = bytearray(
            fixture.packed_weights["q_a_weight"].payload
        )
        self.assertEqual(q_a_payload[2], 127)
        q_a_payload[2] = 126
        payload_inputs = dict(inputs)
        payload_inputs["q_a_weight"] = unpack_q8_0(
            bytes(q_a_payload), (32, 32)
        ).dequantized
        payload_mutation = stage_zero_attention_half(**payload_inputs)
        self.assertEqual(int(np.count_nonzero(
            payload_mutation.q_a != result.q_a
        )), 5)
        self.assertEqual(int(np.count_nonzero(
            payload_mutation.attention_output != result.attention_output
        )), 295)

        ideal_controls = {
            "q_a_weight": ("q_a", 160),
            "q_b_weight": ("q_b", 30720),
            "kv_weight": ("kv_projected", 2560),
            "output_a_weight": ("output_a", 156),
            "output_b_weight": ("output_b", 80),
        }
        for weight_name, (field, expected_count) in ideal_controls.items():
            ideal_inputs = dict(inputs)
            ideal_inputs[weight_name] = fixture.ideal_weights[weight_name]
            ideal_result = stage_zero_attention_half(**ideal_inputs)
            self.assertEqual(
                int(np.count_nonzero(
                    getattr(ideal_result, field) != getattr(result, field)
                )),
                expected_count,
                f"{weight_name} ideal matrix was indistinguishable from payload",
            )

    def test_stage_zero_attention_half_binds_rope_to_cache_frontier(
        self,
    ) -> None:
        for count in (2, 128):
            inputs = self._stage_zero_attention_half_inputs(count)
            for offset in (1, 997):
                bad = dict(inputs)
                bad["absolute_positions"] = (
                    np.asarray(inputs["absolute_positions"], dtype=np.int64)
                    + offset
                )
                with self.assertRaisesRegex(
                    ValueError, "start at committed cache end"
                ):
                    stage_zero_attention_half(**bad)
                with self.assertRaisesRegex(
                    ValueError, "start at committed cache end"
                ):
                    self._mlx_stage_zero_attention_half(bad)

    def test_stage_zero_attention_half_rejects_invalid_inputs_pre_device(
        self,
    ) -> None:
        """Both implementations fail closed before optional MLX is loaded."""

        def nan_copy(value: object) -> np.ndarray:
            result = np.array(value, dtype=np.float32, copy=True)
            result.flat[0] = np.nan
            return result

        def nan_cache(inputs: dict[str, object]) -> object:
            state = inputs["raw_cache"]
            rows = np.array(state.rows, copy=True)
            rows[0, 0, 0] = np.nan
            return type(state)(
                state.capacity, state.token_start, state.length, rows
            )

        def nan_transient(
            inputs: dict[str, object], stage: int
        ) -> np.ndarray:
            result = np.array(
                inputs["other_stage_draft_rows"],
                dtype=np.float32,
                copy=True,
            )
            result[stage, 0, 0] = np.nan
            return result

        cases = (
            ("empty hidden", "hidden_input",
             lambda _i: np.empty((5, 4, 0), dtype=np.float32), {},
             "finite and non-empty"),
            ("nonfinite hidden", "hidden_input",
             lambda i: nan_copy(i["hidden_input"]), {}, "finite"),
            ("HC shape", "hc_function_weight",
             lambda _i: np.zeros((23, 32), dtype=np.float32), {},
             "HC function weight"),
            ("HC scale shape", "hc_scale",
             lambda _i: np.zeros(2, dtype=np.float32), {},
             "HC scale/base"),
            ("HC base shape", "hc_base",
             lambda _i: np.zeros(23, dtype=np.float32), {},
             "HC scale/base"),
            ("attention norm shape", "attention_norm_weight",
             lambda _i: np.zeros(7, dtype=np.float32), {},
             "attention norm"),
            ("Q norm shape", "q_a_norm_weight",
             lambda _i: np.zeros(7, dtype=np.float32), {}, "q_a norm"),
            ("KV shape", "kv_weight",
             lambda _i: np.zeros((511, 8), dtype=np.float32), {},
             "KV weight"),
            ("KV norm shape", "kv_norm_weight",
             lambda _i: np.zeros(511, dtype=np.float32), {}, "KV norm"),
            ("sink shape", "attention_sinks",
             lambda _i: np.zeros(63, dtype=np.float32), {},
             "attention_sinks"),
            ("nonfinite HC", "hc_function_weight",
             lambda i: nan_copy(i["hc_function_weight"]), {}, "finite"),
            ("nonfinite Q", "q_b_weight",
             lambda i: nan_copy(i["q_b_weight"]), {}, "finite"),
            ("nonfinite ring", "raw_cache", nan_cache, {},
             "invalid|finite"),
            ("nonfinite transient stage 1", "other_stage_draft_rows",
             lambda i: nan_transient(i, 1), {}, "finite"),
            ("nonfinite transient stage 2", "other_stage_draft_rows",
             lambda i: nan_transient(i, 2), {}, "finite"),
            ("nonfinite sink", "attention_sinks",
             lambda i: nan_copy(i["attention_sinks"]), {}, "finite"),
            ("nonfinite output A", "output_a_weight",
             lambda i: nan_copy(i["output_a_weight"]), {}, "finite"),
            ("nonfinite output B", "output_b_weight",
             lambda i: nan_copy(i["output_b_weight"]), {}, "finite"),
            ("norm epsilon", None, None, {"norm_eps": 0.0}, "norm_eps"),
            ("HC epsilon", None, None, {"hc_eps": np.nan}, "hc_eps"),
            ("HC iterations zero", None, None, {"hc_iterations": 0},
             "hc_iterations"),
            ("HC iterations bool", None, None, {"hc_iterations": True},
             "hc_iterations"),
            ("RoPE theta", None, None, {"rope_theta": 0.0}, "rope_theta"),
        )

        for label, key, replacement, kwargs, error in cases:
            inputs = self._stage_zero_attention_half_inputs(2)
            if key is not None:
                inputs[key] = replacement(inputs)
            with self.subTest(case=label, implementation="NumPy"):
                with mock.patch.object(
                    mlx_optional,
                    "_mlx",
                    side_effect=AssertionError("device reached"),
                ) as device:
                    with self.assertRaisesRegex(ValueError, error):
                        stage_zero_attention_half(**inputs, **kwargs)
                    device.assert_not_called()
            with self.subTest(case=label, implementation="MLX"):
                with mock.patch.object(
                    mlx_optional,
                    "_mlx",
                    side_effect=AssertionError("device reached"),
                ) as device:
                    with self.assertRaisesRegex(ValueError, error):
                        self._mlx_stage_zero_attention_half(inputs, **kwargs)
                    device.assert_not_called()

    def test_raw_cache_rejects_non_0731_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity must be 128"):
            empty_raw_cache(127, DSPARK_RAW_CACHE_WIDTH)
        with self.assertRaisesRegex(ValueError, "width must be 512"):
            empty_raw_cache(DSPARK_RAW_CACHE_WINDOW, 511)
        case = self.fixture["cases"]["rawCache"]
        partial = prefill_raw_cache(
            self._raw_cache_rows(np.arange(3, 5, dtype=np.int64), case),
            start_position=3,
        )
        with self.assertRaisesRegex(
            ValueError, "partial official raw cache must start"
        ):
            proposal_raw_cache_view(
                partial, 5, self._raw_cache_drafts(case)
            )

    def test_final_head_wires_hc_markov_and_confidence(self) -> None:
        case = self.fixture["cases"]["draftHead"]
        result = finalize_draft_head(
            np.asarray(case["finalHC"], dtype=np.float64),
            case["firstPreviousToken"],
            np.asarray(case["outputProjection"], dtype=np.float64),
            case["normWeight"],
            self._fixture_matrix(case["hcFunction"]),
            case["hcScale"],
            case["hcBase"],
            np.asarray(case["markovEmbedding"], dtype=np.float64),
            np.asarray(case["markovProjection"], dtype=np.float64),
            case["confidenceProjection"],
            confidence_threshold=case["confidenceThreshold"],
        )
        expected = case["expected"]
        np.testing.assert_allclose(result.hidden, expected["hidden"],
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(result.base_logits,
                                   expected["baseLogits"],
                                   rtol=0.0, atol=1e-14)
        self.assertEqual(result.tokens.tolist(), expected["tokens"])
        np.testing.assert_allclose(result.corrected_logits,
                                   expected["correctedLogits"],
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(result.confidence.logits,
                                   expected["confidenceLogits"],
                                   rtol=0.0, atol=1e-14)
        self.assertEqual(result.confidence.keep,
                         expected["confidenceKeep"])

        # These deliberately wrong alternatives prove that the fixture covers
        # final RMSNorm/output and the pre-norm hidden used by confidence.
        collapsed = np.asarray(result.hidden)
        omitted_norm_logits = collapsed @ np.asarray(
            case["outputProjection"], dtype=np.float64
        ).T
        self.assertFalse(np.allclose(omitted_norm_logits, result.base_logits))
        omitted_output = rms_norm(collapsed, case["normWeight"])
        self.assertNotEqual(omitted_output.shape, result.base_logits.shape)
        previous = np.concatenate((
            np.asarray([case["firstPreviousToken"]], dtype=np.int64),
            result.tokens[:-1],
        ))
        wrong_confidence = conditional_confidence(
            rms_norm(collapsed, case["normWeight"]),
            previous,
            np.asarray(case["markovEmbedding"], dtype=np.float64),
            case["confidenceProjection"],
            threshold=case["confidenceThreshold"],
        )
        self.assertFalse(np.allclose(
            wrong_confidence.logits, result.confidence.logits
        ))

    def test_markov_sequential_closed_form_vector(self) -> None:
        # Hand-derived: token 0 selects bias [0, 3] and therefore token 1;
        # feeding that new token into the next step selects [2, 0]. This does
        # not use the fixture generator or another oracle implementation.
        tokens, logits = markov_greedy_draft(
            np.zeros((2, 2), dtype=np.float64),
            0,
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0.0, 2.0], [3.0, 0.0]]),
        )
        self.assertEqual(tokens.tolist(), [1, 0])
        np.testing.assert_array_equal(logits, [[0.0, 3.0], [2.0, 0.0]])

    def test_confidence_prefix_fixture(self) -> None:
        case = self.fixture["cases"]["confidencePrefix"]
        result = conditional_confidence(
            np.asarray(case["hidden"], dtype=np.float64),
            np.asarray(case["previousTokens"], dtype=np.int64),
            np.asarray(case["embedding"], dtype=np.float64),
            np.asarray(case["projection"], dtype=np.float64),
            threshold=case["threshold"],
        )
        np.testing.assert_allclose(
            result.logits,
            np.asarray(case["expected"]["logits"]),
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            result.probabilities,
            np.asarray(case["expected"]["probabilities"]),
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertEqual(result.keep, case["expected"]["keep"])

    def test_confidence_threshold_is_per_position_not_cumulative(self) -> None:
        # Each sigmoid(log(3)) is 3/4, so both positions clear threshold 0.7.
        # A cumulative product would be [3/4, 9/16] and incorrectly keep one.
        result = conditional_confidence(
            np.zeros((2, 1), dtype=np.float64),
            [0, 0],
            np.asarray([[np.log(3.0)]], dtype=np.float64),
            [0.0, 1.0],
            threshold=0.7,
        )
        np.testing.assert_allclose(result.logits, [np.log(3.0)] * 2, rtol=0.0)
        np.testing.assert_allclose(result.probabilities, [0.75, 0.75], rtol=0.0)
        self.assertEqual(result.keep, 2)

    def test_confidence_stops_at_first_low_position(self) -> None:
        # sigmoid(-log(3))=1/4 fails threshold 1/2 immediately.  The later
        # 3/4 cannot reopen a prefix after its first failing position.
        result = conditional_confidence(
            np.zeros((2, 1), dtype=np.float64),
            [0, 1],
            np.asarray([[-np.log(3.0)], [np.log(3.0)]], dtype=np.float64),
            [0.0, 1.0],
            threshold=0.5,
        )
        np.testing.assert_allclose(result.probabilities, [0.25, 0.75], rtol=0.0)
        self.assertEqual(result.keep, 0)

    def test_zero_confidence_threshold_keeps_the_block(self) -> None:
        result = conditional_confidence(
            np.zeros((2, 1), dtype=np.float64),
            [0, 0],
            np.asarray([[-1000.0]], dtype=np.float64),
            [0.0, 1.0],
            threshold=0.0,
        )
        self.assertEqual(result.keep, 2)

    def test_confidence_oracles_have_no_non_production_bias(self) -> None:
        self.assertNotIn("bias", inspect.signature(conditional_confidence).parameters)
        self.assertNotIn(
            "bias", inspect.signature(mlx_optional.confidence_schedule).parameters
        )

    def test_exact_sampling_rejection_fixture(self) -> None:
        case = self.fixture["cases"]["samplingRejected"]
        result = speculative_sample_exact(
            np.asarray(case["targetProbabilities"], dtype=np.float64),
            np.asarray(case["draftTokens"], dtype=np.int64),
            np.asarray(case["draftProbabilities"], dtype=np.float64),
            np.asarray(case["acceptanceUniforms"], dtype=np.float64),
            np.asarray(case["categoricalUniforms"], dtype=np.float64),
        )
        self.assertEqual(result.accepted, case["expected"]["accepted"])
        self.assertEqual(
            result.replacement_token,
            case["expected"]["replacementToken"],
        )
        self.assertEqual(
            list(result.committed_tokens),
            case["expected"]["committedTokens"],
        )
        self.assertEqual(result.target_row, case["expected"]["targetRow"])
        np.testing.assert_allclose(
            result.acceptance_thresholds,
            np.asarray(case["expected"]["acceptanceThresholds"]),
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            result.residual_probabilities,
            np.asarray(case["expected"]["residualProbabilities"]),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_exact_sampling_all_accepted_fixture(self) -> None:
        case = self.fixture["cases"]["samplingAllAccepted"]
        result = speculative_sample_exact(
            np.asarray(case["targetProbabilities"], dtype=np.float64),
            np.asarray(case["draftTokens"], dtype=np.int64),
            np.asarray(case["draftProbabilities"], dtype=np.float64),
            np.asarray(case["acceptanceUniforms"], dtype=np.float64),
            np.asarray(case["categoricalUniforms"], dtype=np.float64),
        )
        self.assertEqual(result.accepted, case["expected"]["accepted"])
        self.assertEqual(
            result.replacement_token,
            case["expected"]["replacementToken"],
        )
        self.assertEqual(
            list(result.committed_tokens),
            case["expected"]["committedTokens"],
        )
        self.assertEqual(result.target_row, case["expected"]["targetRow"])
        np.testing.assert_allclose(
            result.acceptance_thresholds,
            np.asarray(case["expected"]["acceptanceThresholds"]),
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertIsNone(result.residual_probabilities)

    def test_exact_sampling_closed_form_rejection_vector(self) -> None:
        # For drafted token 0, min(1, p/q)=0.2/0.5=0.4. The strict boundary
        # u=0.4 rejects and max(p-q, 0) normalizes from [0, 0.3] to [0, 1].
        result = speculative_sample_exact(
            np.asarray([[0.2, 0.8], [0.5, 0.5]]),
            [0],
            np.asarray([[0.5, 0.5]]),
            [0.4],
            [0.0, 0.0],
        )
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.committed_tokens, (1,))
        np.testing.assert_allclose(result.acceptance_thresholds, [0.4], rtol=0.0)
        np.testing.assert_array_equal(result.residual_probabilities, [0.0, 1.0])

    def test_exact_sampling_closed_form_partial_accept_vector(self) -> None:
        # The first token has p/q >= 1 and commits. At row 1, drafted token 0
        # has threshold 0.2/0.5=0.4; the boundary draw rejects and the residual
        # [0, 0.3] deterministically selects token 1.
        result = speculative_sample_exact(
            np.asarray([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]]),
            [0, 0],
            np.asarray([[0.5, 0.5], [0.5, 0.5]]),
            [0.9, 0.4],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.committed_tokens, (0, 1))
        np.testing.assert_array_equal(result.acceptance_thresholds, [1.0, 0.4])
        np.testing.assert_array_equal(result.residual_probabilities, [0.0, 1.0])

    def test_exact_sampling_closed_form_bonus_vector(self) -> None:
        # For drafted token 1, p/q=0.8/0.5 > 1, so every u in [0,1) accepts.
        # The independent bonus row [0.75, 0.25] and u=0.9 select token 1.
        result = speculative_sample_exact(
            np.asarray([[0.2, 0.8], [0.75, 0.25]]),
            [1],
            np.asarray([[0.5, 0.5]]),
            [0.999],
            [0.0, 0.9],
        )
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.committed_tokens, (1, 1))
        np.testing.assert_array_equal(result.acceptance_thresholds, [1.0])
        self.assertIsNone(result.residual_probabilities)

    def test_sampling_matches_deepspec_numeric_floors(self) -> None:
        # The pinned evaluator clamps selected q to 1e-8.  Here p=5e-9 and
        # q=0 therefore produce threshold 1/2.  Rejection leaves only 5e-9 of
        # positive p-q mass, so the <=1e-8 guard falls back to the target row;
        # u=1/2 then selects token 1 rather than the raw residual's token 0.
        target = np.asarray([[5.0e-9, 1.0 - 5.0e-9], [0.5, 0.5]])
        result = speculative_sample_exact(
            target,
            [0],
            np.asarray([[0.0, 1.0]]),
            [0.5],
            [0.5, 0.0],
        )
        np.testing.assert_allclose(result.acceptance_thresholds, [0.5], rtol=0.0)
        np.testing.assert_allclose(
            result.residual_probabilities,
            target[0],
            rtol=0.0,
            atol=1.0e-16,
        )
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.committed_tokens, (1,))

    def test_0731_metadata_contract(self) -> None:
        metadata = self.fixture["cases"]["metadata0731"]
        hf = validate_0731_metadata(metadata["hf"], flavor="hf")
        gguf = validate_0731_metadata(metadata["gguf"], flavor="gguf")
        support = validate_0731_metadata(
            metadata["support"], flavor="support"
        )
        self.assertEqual(hf.block_size, 5)
        self.assertEqual(hf.stage_count, 3)
        self.assertEqual(hf.hc_lanes, 4)
        self.assertEqual(hf.sinkhorn_iterations, 20)
        self.assertEqual(hf.raw_cache_window, DSPARK_RAW_CACHE_WINDOW)
        self.assertEqual(hf.raw_cache_width, DSPARK_RAW_CACHE_WIDTH)
        self.assertEqual(hf, gguf)
        self.assertEqual(hf, support)

    def test_exact_three_stage_support_tensor_schema(self) -> None:
        expected = expected_tensor_schema()
        self.assertEqual(len(expected), 81)
        self.assertEqual(
            [sum(name.startswith(f"mtp.{stage}.") for name in expected)
             for stage in range(3)],
            [26, 24, 31],
        )
        self.assertIn("mtp.0.main_proj.weight", expected)
        self.assertNotIn("mtp.1.main_proj.weight", expected)
        self.assertIn("mtp.2.confidence_head.proj.weight", expected)
        self.assertEqual(
            expected["mtp.0.main_proj.weight"].dimensions,
            (12288, 4096),
        )
        self.assertEqual(
            expected["mtp.2.confidence_head.proj.weight"].dimensions,
            (4352, 1),
        )

    def test_synthetic_support_header_round_trip_and_fail_closed_schema(self) -> None:
        metadata = self.fixture["cases"]["metadata0731"]["support"]
        expected = expected_tensor_schema()
        encoded = _synthetic_support_gguf(metadata, expected)
        with tempfile.TemporaryDirectory(prefix="dspark-header-") as tmp:
            path = Path(tmp) / "support.gguf"
            path.write_bytes(encoded)
            header = read_support_header(path)
        self.assertEqual(header.version, 3)
        self.assertEqual(header.tensors, expected)
        self.assertEqual(validate_support_header(header).stage_count, 3)

        malformed = dict(expected)
        malformed["mtp.1.attn_kv.weight"] = TensorSpec(8, (4096, 511))
        with self.assertRaisesRegex(
                SupportSchemaError, "descriptor mismatches"):
            validate_support_header(SupportHeader(3, metadata, malformed))

        encoded_v2 = _synthetic_support_gguf(metadata, expected, version=2)
        with tempfile.TemporaryDirectory(prefix="dspark-header-v2-") as tmp:
            path = Path(tmp) / "support-v2.gguf"
            path.write_bytes(encoded_v2)
            with self.assertRaisesRegex(
                    SupportSchemaError, "requires GGUF version 3"):
                read_support_header(path)

    def test_support_layout_rejects_overlap_alignment_and_bounds(self) -> None:
        tensors = {
            "a": TensorSpec(0, (1,)),
            "b": TensorSpec(0, (1,)),
        }
        metadata = {"general.alignment": 32}
        valid = SupportHeader(
            3, metadata, tensors, {"a": 0, "b": 32}, header_end=32
        )
        validate_support_layout(valid, file_size=96)

        overlap = SupportHeader(
            3, metadata, tensors, {"a": 0, "b": 0}, header_end=32
        )
        with self.assertRaisesRegex(SupportSchemaError, "payload overlap"):
            validate_support_layout(overlap, file_size=64)
        unaligned = SupportHeader(
            3, metadata, tensors, {"a": 0, "b": 4}, header_end=32
        )
        with self.assertRaisesRegex(SupportSchemaError, "unaligned"):
            validate_support_layout(unaligned, file_size=64)
        with self.assertRaisesRegex(SupportSchemaError, "beyond end"):
            validate_support_layout(valid, file_size=64)

    @unittest.skipUnless(
        os.environ.get("DS4_DSPARK_SUPPORT_GGUF"),
        "set DS4_DSPARK_SUPPORT_GGUF for header-only final artifact validation",
    )
    def test_optional_final_support_header(self) -> None:
        path = Path(os.environ["DS4_DSPARK_SUPPORT_GGUF"])
        semantic = validate_support_file(path)
        self.assertEqual(semantic.stage_count, 3)
        self.assertEqual(semantic.target_layer_ids, (40, 41, 42))

    def test_0731_metadata_fails_closed(self) -> None:
        metadata = dict(self.fixture["cases"]["metadata0731"]["gguf"])
        metadata["dspark.target_layer_ids"] = [40, 42, 41]
        with self.assertRaisesRegex(MetadataError, "target_layer_ids"):
            validate_0731_metadata(metadata, flavor="gguf")

    def test_0731_gguf_stage_records_agree(self) -> None:
        metadata = dict(self.fixture["cases"]["metadata0731"]["gguf"])
        metadata["dspark.n_layers"] = 2
        with self.assertRaisesRegex(MetadataError, "dspark.n_layers"):
            validate_0731_metadata(metadata, flavor="gguf")

    def test_0731_metadata_rejects_aliases(self) -> None:
        metadata = dict(self.fixture["cases"]["metadata0731"]["gguf"])
        metadata["deepseek4.dspark.block_size"] = 5
        with self.assertRaisesRegex(MetadataError, "unknown DSpark keys"):
            validate_0731_metadata(metadata, flavor="gguf")
        metadata = dict(self.fixture["cases"]["metadata0731"]["support"])
        metadata["DSpark.Source.Revision"] = metadata[
            "dspark.source.revision"
        ]
        with self.assertRaisesRegex(MetadataError, "unknown DSpark keys"):
            validate_0731_metadata(metadata, flavor="support")

    def test_pinned_provenance(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schemaVersion"], 3)
        self.assertFalse(provenance["implementation"]["productionDependency"])
        self.assertFalse(provenance["implementation"]["externalRuntimeCodeCopied"])
        self.assertEqual(
            provenance["implementation"]["supportSchemaDerivation"]["status"],
            "pending",
        )
        self.assertEqual(
            provenance["implementation"]["runtimeCaptureParity"]["status"],
            "pending",
        )
        ffn_fixture = provenance["implementation"]["ffnFixture"]
        self.assertEqual(ffn_fixture["version"], 2)
        self.assertEqual(
            ffn_fixture["sharedGateQ8Sha256"],
            "7f10192cdc295ae811318df8d95457b4a1d45aed92e12c5cda99cafa82772e6e",
        )
        self.assertEqual(
            ffn_fixture["sharedUpQ8Sha256"],
            "24e79016a69fc0472707508c4536b5c4a909bacee2dbcb37401e0a94737b2a1f",
        )
        self.assertEqual(
            ffn_fixture["sharedDownQ8Sha256"],
            "71f5a2c1cc0487b8391aac802e2cddbe127d011bff77d20bc59d2e0470117a9d",
        )
        self.assertEqual(
            ffn_fixture["systemValidation"]["result"],
            "Ran 58 tests; OK with 2 explicit optional MLX/support-file skips",
        )
        self.assertEqual(ffn_fixture["mlxValidation"]["status"], "passed")
        self.assertEqual(
            ffn_fixture["mlxValidation"]["validatedFixtureVersion"], 2
        )
        target = next(
            source
            for source in provenance["sources"]
            if source["role"] == "official-target-config"
        )
        self.assertEqual(
            target["revision"], "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        )
        self.assertEqual(
            target["sha256"],
            "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        )
        index = next(
            source
            for source in provenance["sources"]
            if source["role"] == "official-target-index"
        )
        self.assertEqual(index["observations"]["mtpTensorNames"], 4_705)
        self.assertEqual(
            [item["path"] for item in index["observations"]["mtpSourceShards"]],
            [
                "model-00046-of-00048.safetensors",
                "model-00047-of-00048.safetensors",
                "model-00048-of-00048.safetensors",
            ],
        )
        support = next(
            source
            for source in provenance["sources"]
            if source["role"] == "reference-quantized-support-gguf"
        )
        self.assertEqual(
            support["revision"],
            "2286cd2429a0f2400ac84b07e9a2274114db9dd2",
        )
        self.assertEqual(support["size"], 5_989_114_272)
        self.assertEqual(
            support["sha256"],
            "8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e",
        )
        self.assertFalse(support["publicationAccepted"])
        self.assertEqual(
            support["quantizer"]["revision"],
            "fc9efd1cc7550d47c497a2c6ba0eeb85baa4df0f",
        )
        final_support = next(
            source
            for source in provenance["sources"]
            if source["role"] == "authenticated-final-support-gguf"
        )
        self.assertEqual(final_support["size"], 5_989_114_912)
        self.assertEqual(
            final_support["sha256"],
            "aa2bd4b5b916e1aa0a01392d69cbdd9798a3f3050c29c22973c8ee4233af0413",
        )
        self.assertTrue(final_support["sourceAccepted"])
        self.assertFalse(final_support["publicationAccepted"])
        self.assertEqual(
            final_support["observations"]["stageTensorCounts"],
            [26, 24, 31],
        )
        equations = next(
            source
            for source in provenance["sources"]
            if source["role"] == "official-dspark-inference-equations"
        )
        self.assertEqual(
            {item["path"]: item["sha256"] for item in equations["files"]},
            {
                "inference/config.json":
                    "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71",
                "inference/model.py":
                    "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
                "inference/kernel.py":
                    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
            },
        )
        algorithm = next(
            source
            for source in provenance["sources"]
            if source["role"] == "algorithm-definition"
        )
        self.assertEqual(
            algorithm["revision"],
            "005e03b81cec38b7da6399833d609ee89a2587f2",
        )
        self.assertEqual(
            {
                item["path"]: item["sha256"]
                for item in algorithm["files"]
            },
            {
                "deepspec/eval/dspark/draft_ops.py":
                    "9d07a301fa643ee3b558a093647ff3db2918a47c96267d3546081eb9df44b799",
                "deepspec/modeling/dspark/markov_head.py":
                    "6659bcdc12d923d4fc16cc2280c03078c2110a4792305e1f3f42b5468f75ef46",
                "deepspec/eval/base_evaluator.py":
                    "f7630a0da6a4c6ee370ebe66e2cccf3a803c5331ea9ebf6f54b2285cc3b435a9",
                "deepspec/utils/sampling.py":
                    "8f8ff58387e70526452672372628d1b2fbacb778a89f1a46abc9e2b7639d215a",
            },
        )

    def test_mlx_float32_limits_are_operation_specific(self) -> None:
        self.assertEqual(
            mlx_optional.MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT,
            1.0e-4,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_CONFIDENCE_MAX_ABS_DRIFT,
            5.0e-8,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_HC_MEAN_MAX_ABS_DRIFT,
            1.0e-7,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_MAIN_PROJECTION_MAX_ABS_DRIFT,
            5.0e-8,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_RAW_CONTEXT_MAIN_MAX_ABS_DRIFT,
            1.0e-7,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_HC_SPLIT_MAX_ABS_DRIFT,
            1.0e-7,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT,
            5.0e-7,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_CONTEXT_PROJECTED_MAX_ABS_DRIFT,
            3.90625e-3,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_CONTEXT_NORMALIZED_MAX_ABS_DRIFT,
            7.8125e-3,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_CONTEXT_ROPE_MAX_ABS_DRIFT,
            7.8125e-3,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_CONTEXT_STORED_MAX_ABS_DRIFT,
            6.25e-2,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_CONTEXT_SCALE_MAX_ABS_DRIFT,
            0.0,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_ATTENTION_OUTPUT_MAX_ABS_DRIFT,
            2.44140625e-4,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT,
            2.44140625e-4,
        )
        self.assertEqual(
            mlx_optional.MLX_BF16_PHYSICAL_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT,
            1.220703125e-4,
        )
        self.assertEqual(
            mlx_optional.MLX_FFN_OPERATION_MAX_ABS_DRIFT,
            {
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
            },
        )
        self.assertGreater(
            mlx_optional.MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT,
            mlx_optional.MLX_F32_CONFIDENCE_MAX_ABS_DRIFT,
        )

    def test_physical_ffn_payload_geometry_and_codecs(self) -> None:
        fixture = build_physical_ffn_fixture()
        manifest = ffn_payload_manifest(fixture)
        self.assertEqual(manifest["fixture_version"], 2)
        self.assertEqual(manifest["geometry"], {
            "proposal_rows": 5,
            "hc_lanes": 4,
            "hidden_width": 4096,
            "expert_count": 256,
            "topk": 6,
            "compact_mid_width": 256,
        })
        self.assertEqual(manifest["selected_experts"], FFN_SELECTED_EXPERTS.tolist())
        self.assertEqual(manifest["routed_payload_bytes"], 26_542_080)
        self.assertEqual(
            manifest["routed_payload_sha256"],
            "49bf413cc3e98472a7aceb53b1b4a16b83ab2182354dfe766815c3a064f274bb",
        )
        self.assertEqual(len(fixture.routed_expert_weights), 30)
        self.assertEqual(
            set(fixture.routed_expert_weights),
            set(int(item) for item in FFN_SELECTED_EXPERTS.reshape(-1)),
        )
        self.assertTrue({0, 29, 30, 255}.issubset(
            fixture.routed_expert_weights
        ))

        for expert in (0, 29, 30, 255):
            record = fixture.routed_expert_weights[expert]
            self.assertEqual(
                [record[role].storage for role in ("gate", "up", "down")],
                ["IQ2_XXS", "IQ2_XXS", "Q2_K"],
            )
            self.assertEqual(record["gate"].logical_shape, (256, 4096))
            self.assertEqual(record["up"].logical_shape, (256, 4096))
            self.assertEqual(record["down"].logical_shape, (4096, 256))
            self.assertFalse(hasattr(record["gate"], "ideal"))
            gate = record["gate"].dequantized()
            up = record["up"].dequantized()
            down = record["down"].dequantized()
            self.assertEqual(gate.shape, (256, 4096))
            self.assertEqual(down.shape, (4096, 256))
            self.assertGreater(np.unique(gate).size, 8)
            self.assertGreater(np.unique(down).size, 8)
            self.assertGreater(int(np.count_nonzero(gate != up)), 100_000)

        self.assertNotEqual(
            fixture.routed_expert_weights[0]["gate"].payload,
            fixture.routed_expert_weights[29]["gate"].payload,
        )
        self.assertEqual(
            manifest["non_routed"]["shared_gate_weight"]["sha256"],
            "7f10192cdc295ae811318df8d95457b4a1d45aed92e12c5cda99cafa82772e6e",
        )
        self.assertEqual(
            manifest["non_routed"]["shared_up_weight"]["sha256"],
            "24e79016a69fc0472707508c4536b5c4a909bacee2dbcb37401e0a94737b2a1f",
        )
        self.assertEqual(
            manifest["non_routed"]["shared_down_weight"]["sha256"],
            "71f5a2c1cc0487b8391aac802e2cddbe127d011bff77d20bc59d2e0470117a9d",
        )
        gate_blocks = np.frombuffer(
            fixture.packed_weights["shared_gate_weight"].payload,
            dtype=np.uint8,
        ).reshape(-1, 34)
        self.assertEqual(
            np.ascontiguousarray(gate_blocks[0, :2]).view("<f2")[0],
            np.float16(1.0 / 256.0),
        )
        self.assertEqual(gate_blocks[0, 2:].view(np.int8)[0], -127)
        down_blocks = np.frombuffer(
            fixture.packed_weights["shared_down_weight"].payload,
            dtype=np.uint8,
        ).reshape(4096, 8, 34)
        self.assertEqual(
            np.ascontiguousarray(down_blocks[67, 0, :2]).view("<f2")[0],
            np.float16(1.0),
        )
        self.assertEqual(down_blocks[67, 0, 2:].view(np.int8)[0], 127)
        self.assertEqual(int(np.count_nonzero(down_blocks[67])), 2)
        broken_iq = bytearray(
            fixture.routed_expert_weights[0]["gate"].payload
        )
        broken_iq[2] = 1
        with self.assertRaisesRegex(ValueError, "grid index zero"):
            unpack_iq2_xxs(bytes(broken_iq), (256, 4096))
        broken_q2 = bytearray(
            fixture.routed_expert_weights[0]["down"].payload
        )
        original_q2 = unpack_q2_k(bytes(broken_q2), (4096, 256))
        broken_q2[16] ^= 0x03
        mutated_q2 = unpack_q2_k(bytes(broken_q2), (4096, 256))
        self.assertGreater(int(np.count_nonzero(original_q2 != mutated_q2)), 0)

    def test_physical_ffn_freezes_official_bf16_and_routing_seams(self) -> None:
        fixture = build_physical_ffn_fixture()
        result = dspark_reference.stage_ffn_moe_payload_first(**fixture.inputs)
        np.testing.assert_array_equal(
            result.selected_experts, FFN_SELECTED_EXPERTS
        )
        self.assertEqual(np.unique(result.selected_experts).size, 30)
        selection_scores = (
            result.router_probabilities
            + np.asarray(
                fixture.inputs["selection_bias"], dtype=np.float32
            )[None, :]
        )
        expert_ids = np.arange(256, dtype=np.int32)
        cutoff_margins = []
        for row in selection_scores:
            order = np.lexsort((expert_ids, -row))
            cutoff_margins.append(float(row[order[5]] - row[order[6]]))
        self.assertEqual(min(cutoff_margins), 0.04725050926208496)
        self.assertGreater(
            min(cutoff_margins),
            2.0 * max(
                mlx_optional.MLX_FFN_OPERATION_MAX_ABS_DRIFT[
                    "router_logits"
                ],
                mlx_optional.MLX_FFN_OPERATION_MAX_ABS_DRIFT[
                    "router_probabilities"
                ],
            ),
        )
        np.testing.assert_allclose(
            np.sum(result.expert_weights, axis=1, dtype=np.float32),
            np.full(5, 1.5, dtype=np.float32),
            rtol=0.0,
            atol=1.1920928955078125e-7,
        )
        self.assertGreater(np.unique(result.expert_weights).size, 12)

        bf16_fields = (
            "hidden_input", "hc_pre_output", "ffn_normalized",
            "shared_gate", "shared_up", "shared_mid", "shared_down",
            "routed_gate", "routed_up", "routed_mid", "routed_down",
            "moe_output", "hc_post_output",
        )
        for field in bf16_fields:
            self._assert_bfloat16_boundary(getattr(result, field))
        self.assertGreater(
            int(np.count_nonzero(
                result.router_probabilities
                != self._round_fixture_bfloat16(result.router_probabilities)
            )),
            1_000,
        )
        self.assertGreater(
            int(np.count_nonzero(
                result.routed_sum
                != self._round_fixture_bfloat16(result.routed_sum)
            )),
            10_000,
        )

        expected_shapes = {
            "ffn_normalized": (5, 4096),
            "router_logits": (5, 256),
            "selected_experts": (5, 6),
            "shared_mid": (5, 256),
            "routed_mid": (5, 6, 256),
            "routed_down": (5, 6, 4096),
            "moe_output": (5, 4096),
            "hc_post_output": (5, 4, 4096),
        }
        for field, shape in expected_shapes.items():
            self.assertEqual(getattr(result, field).shape, shape, field)

        expected_digests = {
            "ffn_normalized": "c6a60e7d9980460fd4e0c7c0bf86e3714529c97985a85d7bcaafede9d28660f3",
            "router_probabilities":
                "c0348ed01aff47a3c2848dd67b5e5acddb2ffd4323e1443d5f9d4f018568b63a",
            "expert_weights": "895dc2d1c4a85c9287b8b54650f2cd6594eca9c9bba28c6bac3c6e5ccc657838",
            "shared_gate": "da153e59f47374f0c7f8e5f4b85ad48449a12ae62e228422399e6de125471be5",
            "shared_up": "ee06aa93ae38f5e1ef7fd566738a0065f0f5ecc90799d354d7d7cdaf7a6144c2",
            "shared_mid": "8308c7c0efd0eedb71a89a7fb34484ae54547ec4e12700924b47fc251deded49",
            "shared_down": "415e3899f0732575d7f3f6e1947411db19a4009f67c74f5ef45324042bbd9ffa",
            "routed_mid": "0d97fe47189317749e687e17b936557b63e0876d3882723644d2fe115957c107",
            "routed_sum": "a0fffc630dffb5e5453e4a592b3259cb518a1de0179c7565ec3639674af1ad85",
            "moe_output": "09e15e7ce1d65c2a7d31d1a9a77791498845eca6da17189869bb556cba088dd6",
            "hc_post_output": "9e66222140513bf2a12f5f4ee256bf82ae425b0114e073e7c0fd073cdcd1489d",
        }
        for field, digest in expected_digests.items():
            self.assertEqual(
                self._array_digest(getattr(result, field), "<f4"),
                digest,
                field,
            )

    def test_physical_ffn_mutations_pin_bias_ties_clamp_and_sum_order(self) -> None:
        fixture = build_physical_ffn_fixture()
        result = dspark_reference.stage_ffn_moe_payload_first(**fixture.inputs)

        zero_bias = np.zeros(256, dtype=np.float32)
        unbiased = dspark_reference.dspark_router_q8(
            result.ffn_normalized, fixture.inputs["router_weight"], zero_bias
        )
        self.assertFalse(np.array_equal(
            unbiased.selected_experts, result.selected_experts
        ))
        for row in range(5):
            self.assertEqual(
                set(unbiased.selected_experts[row]),
                set(result.selected_experts[row]),
            )
            base_by_id = {
                int(expert): float(result.expert_weights[row, slot])
                for slot, expert in enumerate(result.selected_experts[row])
            }
            unbiased_by_id = {
                int(expert): float(unbiased.expert_weights[row, slot])
                for slot, expert in enumerate(unbiased.selected_experts[row])
            }
            for expert in base_by_id:
                self.assertAlmostEqual(
                    base_by_id[expert], unbiased_by_id[expert], places=6
                )
        np.testing.assert_array_equal(
            unbiased.probabilities, result.router_probabilities
        )

        tied_router = np.array(
            fixture.inputs["router_weight"].dequantized, copy=True
        )
        tied_router[[0, 1, 2, 3, 4, 5, 25]] = tied_router[0]
        tied = dspark_reference.dspark_router_q8(
            result.ffn_normalized, tied_router, zero_bias
        )
        np.testing.assert_array_equal(
            tied.selected_experts[0], np.arange(6, dtype=np.int32)
        )
        self.assertNotIn(25, tied.selected_experts[0])

        duplicated_input = np.array(result.ffn_normalized, copy=True)
        duplicated_input[1] = duplicated_input[0]
        duplicated = dspark_reference.dspark_router_q8(
            duplicated_input,
            fixture.inputs["router_weight"],
            fixture.inputs["selection_bias"],
        )
        np.testing.assert_array_equal(
            duplicated.selected_experts[1], duplicated.selected_experts[0]
        )
        self.assertEqual(np.unique(duplicated.selected_experts).size, 24)
        duplicated_stage_inputs = dict(fixture.inputs)
        duplicated_hidden = np.array(
            duplicated_stage_inputs["hidden_input"], copy=True
        )
        duplicated_hidden[1] = duplicated_hidden[0]
        duplicated_stage_inputs["hidden_input"] = duplicated_hidden
        duplicated_stage = dspark_reference.stage_ffn_moe_payload_first(
            **duplicated_stage_inputs
        )
        np.testing.assert_array_equal(
            duplicated_stage.selected_experts[1],
            duplicated_stage.selected_experts[0],
        )
        self.assertEqual(
            np.unique(duplicated_stage.selected_experts).size, 24
        )

        unclamped = dspark_reference.stage_ffn_moe_payload_first(
            **fixture.inputs, swiglu_clamp=0.0
        )
        self.assertGreater(np.max(np.abs(result.routed_gate)), 10.0)
        self.assertGreater(np.max(np.abs(result.routed_up)), 10.0)
        self.assertGreater(
            int(np.count_nonzero(result.shared_mid != unclamped.shared_mid)), 0
        )
        self.assertGreater(
            int(np.count_nonzero(result.routed_mid != unclamped.routed_mid)),
            1_000,
        )
        self.assertGreater(
            int(np.count_nonzero(result.hc_post_output !=
                                 unclamped.hc_post_output)),
            10_000,
        )

        original_swiglu = dspark_reference._swiglu_f32

        def symmetric_gate_clamp(
            gate: np.ndarray,
            up: np.ndarray,
            *,
            clamp: float,
        ) -> np.ndarray:
            wrong_gate = np.asarray(gate, dtype=np.float32)
            if clamp > 1.0e-6:
                wrong_gate = np.maximum(
                    wrong_gate, np.float32(-clamp)
                )
            return original_swiglu(wrong_gate, up, clamp=clamp)

        with mock.patch.object(
            dspark_reference,
            "_swiglu_f32",
            side_effect=symmetric_gate_clamp,
        ):
            symmetric = dspark_reference.stage_ffn_moe_payload_first(
                **fixture.inputs
            )
        clamp_controls = {
            "shared_mid": (1, 0.004547107499092817),
            "shared_down": (16, 0.5781235694885254),
            "moe_output": (1, 0.5),
            "hc_post_output": (4, 0.75),
        }
        for field, (different, maximum) in clamp_controls.items():
            delta = np.abs(
                getattr(result, field) - getattr(symmetric, field)
            )
            self.assertEqual(int(np.count_nonzero(delta)), different, field)
            self.assertEqual(float(np.max(delta)), maximum, field)

        slot_order_sum = np.zeros_like(result.routed_sum)
        for row in range(5):
            for slot in range(6):
                slot_order_sum[row] += result.routed_down[row, slot]
        self.assertEqual(
            int(np.count_nonzero(slot_order_sum != result.routed_sum)),
            6_144,
        )
        self.assertEqual(
            float(np.max(np.abs(slot_order_sum - result.routed_sum))),
            1.9073486328125e-6,
        )

    def test_dspark_router_fails_closed_per_row_on_f32_extremes(self) -> None:
        bias = np.zeros(256, dtype=np.float32)
        normalized = np.zeros((5, 32), dtype=np.float32)
        normalized[0] = 1.0
        underflow_router = np.full((256, 32), -32.0, dtype=np.float32)
        underflow = dspark_reference.dspark_router_q8(
            normalized, underflow_router, bias
        )
        self.assertTrue(np.all(np.isfinite(underflow.logits)))
        np.testing.assert_array_equal(
            underflow.selected_experts[0],
            np.full(6, -1, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            underflow.probabilities[0], np.zeros(256, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            underflow.expert_weights[0], np.zeros(6, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            underflow.selected_experts[1:],
            np.tile(np.arange(6, dtype=np.int32), (4, 1)),
        )
        np.testing.assert_allclose(
            underflow.expert_weights[1:],
            np.full((4, 6), 0.25, dtype=np.float32),
            rtol=0.0,
            atol=1.4901161193847656e-8,
        )

        maximum = np.finfo(np.float32).max
        overflow_router = np.full((256, 32), maximum, dtype=np.float32)
        overflow_source = np.zeros((5, 32), dtype=np.float32)
        overflow_source[0] = maximum
        overflow = dspark_reference.dspark_router_q8(
            overflow_source, overflow_router, bias
        )
        self.assertFalse(np.all(np.isfinite(overflow.logits[0])))
        np.testing.assert_array_equal(
            overflow.selected_experts[0],
            np.full(6, -1, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            overflow.probabilities[0], np.zeros(256, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            overflow.expert_weights[0], np.zeros(6, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            overflow.selected_experts[1:],
            np.tile(np.arange(6, dtype=np.int32), (4, 1)),
        )

    def test_native_support_routed_checkpoint_matches_frozen_oracle(self) -> None:
        source = DS4_METAL_SOURCE_PATH.read_text(encoding="utf-8")

        def initializer(name: str) -> str:
            match = re.search(
                rf"{re.escape(name)}[^=]*=\s*\{{(.*?)\n\s*\}};",
                source,
                re.DOTALL,
            )
            self.assertIsNotNone(match, name)
            assert match is not None
            return match.group(1)

        def uint32_values(name: str) -> np.ndarray:
            return np.asarray([
                int(value, 16)
                for value in re.findall(
                    r"UINT32_C\(0x([0-9a-fA-F]+)\)",
                    initializer(name),
                )
            ], dtype=np.uint32)

        selected = np.asarray([
            int(value)
            for value in re.findall(
                r"(?<![xA-Fa-f0-9])(-?\d+)(?=\s*,)",
                initializer("g_dspark_routed_test_selected"),
            )
        ], dtype=np.int32).reshape(5, 6)
        np.testing.assert_array_equal(selected, FFN_SELECTED_EXPERTS)

        fixture = build_physical_ffn_fixture()
        result = dspark_reference.stage_ffn_moe_payload_first(
            **fixture.inputs
        )
        weight_bits = uint32_values(
            "g_dspark_routed_test_weight_bits"
        )
        np.testing.assert_array_equal(
            weight_bits,
            result.expert_weights.astype("<f4").view("<u4").reshape(-1),
        )

        def in_expert_id_order(field: str) -> np.ndarray:
            values = getattr(result, field)
            return np.stack([
                values[row, slot]
                for row in range(5)
                for slot in np.argsort(
                    result.selected_experts[row], kind="stable"
                )
            ]).astype("<f4")

        sorted_down = in_expert_id_order("routed_down").view("<u4")
        down_pattern = uint32_values(
            "g_dspark_routed_test_down_pattern"
        ).reshape(30, 4)
        np.testing.assert_array_equal(down_pattern, sorted_down[:, :4])
        np.testing.assert_array_equal(
            sorted_down, np.tile(down_pattern, (1, 1024))
        )

        routed_sum = result.routed_sum.astype("<f4").view("<u4")
        sum_pattern = uint32_values(
            "g_dspark_routed_test_sum_pattern"
        ).reshape(5, 4)
        np.testing.assert_array_equal(sum_pattern, routed_sum[:, :4])
        np.testing.assert_array_equal(
            routed_sum, np.tile(sum_pattern, (1, 1024))
        )

        expected_hashes = {
            "input": self._array_digest(result.ffn_normalized, "<f4"),
            "gate": self._array_digest(
                in_expert_id_order("routed_gate"), "<f4"
            ),
            "up": self._array_digest(
                in_expert_id_order("routed_up"), "<f4"
            ),
            "mid": self._array_digest(
                in_expert_id_order("routed_mid"), "<f4"
            ),
            "down": self._array_digest(
                in_expert_id_order("routed_down"), "<f4"
            ),
            "sum": self._array_digest(result.routed_sum, "<f4"),
        }
        self.assertEqual(expected_hashes, {
            "input":
                "c6a60e7d9980460fd4e0c7c0bf86e3714529c97985a85d7bcaafede9d28660f3",
            "gate":
                "5fab35c8aada22905d435ee298f2cb8b2ee50e46db0bbfc43aee6d16e5dd9f9c",
            "up":
                "b9599f7561f30bd39563f2869c3b29d2d8a24ec04da4cbbc29ac27055ba6d5cd",
            "mid":
                "121738d8b17fd4709825df9eda1817067f880fe2692158d527b616bcf401c6b2",
            "down":
                "c851ef3f25fb561eda6dd6c8d3cf228cf07ac9ca121df1237e99ed18a5eca30a",
            "sum":
                "a0fffc630dffb5e5453e4a592b3259cb518a1de0179c7565ec3639674af1ad85",
        })
        for digest in expected_hashes.values():
            self.assertIn(digest, source)
        self.assertEqual(
            ffn_payload_manifest(fixture)["routed_payload_sha256"],
            "49bf413cc3e98472a7aceb53b1b4a16b83ab2182354dfe766815c3a064f274bb",
        )
        self.assertIn(
            "49bf413cc3e98472a7aceb53b1b4a16b83ab2182354dfe766815c3a064f274bb",
            source,
        )

        manifest = ffn_payload_manifest(fixture)
        shared_payloads = [
            fixture.packed_weights[name].payload
            for name in (
                "shared_gate_weight",
                "shared_up_weight",
                "shared_down_weight",
            )
        ]
        self.assertEqual(
            hashlib.sha256(b"".join(shared_payloads)).hexdigest(),
            "0016bbdbc7ff6800342be3a2ad8ab209290f04fa156774aa169d9f2798ef0b62",
        )
        for name, digest in (
            ("shared_gate_weight",
             "7f10192cdc295ae811318df8d95457b4a1d45aed92e12c5cda99cafa82772e6e"),
            ("shared_up_weight",
             "24e79016a69fc0472707508c4536b5c4a909bacee2dbcb37401e0a94737b2a1f"),
            ("shared_down_weight",
             "71f5a2c1cc0487b8391aac802e2cddbe127d011bff77d20bc59d2e0470117a9d"),
        ):
            self.assertEqual(manifest["non_routed"][name]["sha256"], digest)
            self.assertIn(digest, source)

        _, split = hc_pre(
            result.hidden_input,
            fixture.packed_weights["hc_ffn_function_weight"].dequantized,
            fixture.inputs["hc_scale"],
            fixture.inputs["hc_base"],
        )
        packed_split = np.concatenate((
            split.pre,
            split.post,
            split.combination.reshape(5, 16),
        ), axis=1).astype("<f4")
        native_split = uint32_values(
            "g_dspark_shared_test_hc_split_bits"
        ).reshape(5, 24)
        np.testing.assert_array_equal(native_split, packed_split.view("<u4"))
        self.assertEqual(
            self._array_digest(packed_split, "<f4"),
            "d7901dc0f69350252a7eee61a95bacaccb65bfac38f1adfb300ff5debb996ebc",
        )

        shared_hashes = {
            "shared_gate":
                "da153e59f47374f0c7f8e5f4b85ad48449a12ae62e228422399e6de125471be5",
            "shared_up":
                "ee06aa93ae38f5e1ef7fd566738a0065f0f5ecc90799d354d7d7cdaf7a6144c2",
            "shared_mid":
                "8308c7c0efd0eedb71a89a7fb34484ae54547ec4e12700924b47fc251deded49",
            "shared_down":
                "415e3899f0732575d7f3f6e1947411db19a4009f67c74f5ef45324042bbd9ffa",
            "moe_output":
                "09e15e7ce1d65c2a7d31d1a9a77791498845eca6da17189869bb556cba088dd6",
            "hidden_input":
                "bf0c4d85aeea6ccf4a50627856bd3fe0d39344176a48833f35539df57bfe89e6",
            "hc_post_output":
                "9e66222140513bf2a12f5f4ee256bf82ae425b0114e073e7c0fd073cdcd1489d",
        }
        for field, digest in shared_hashes.items():
            self.assertEqual(
                self._array_digest(getattr(result, field), "<f4"),
                digest,
                field,
            )
            self.assertIn(digest, source)

        encode_start = source.index(
            "static int ds4_gpu_dspark_routed_test_encode("
        )
        encode_end = source.index(
            "static int ds4_gpu_dspark_routed_test_shadow_unchanged(",
            encode_start,
        )
        encode_source = source[encode_start:encode_end]
        self.assertEqual(encode_source.count("ds4_gpu_matmul_q8_0_tensor("), 3)
        self.assertEqual(
            encode_source.count("ds4_gpu_encode_moe_swiglu_weight("), 2
        )
        self.assertIn("ds4_gpu_add_tensor(", encode_source)
        self.assertIn("ds4_gpu_hc_expand_split_tensor(", encode_source)
        self.assertIn("DS4_DSPARK_ROUTED_TEST_FAIL_AFTER_SHARED_DOWN", source)
        self.assertIn("DS4_DSPARK_ROUTED_TEST_FAIL_AFTER_MOE", source)

        test_start = source.index(
            "int ds4_gpu_internal_dspark_support_routed_moe_test(void)"
        )
        test_end = source.index(
            "enum {\n    DS4_DSPARK_HISTORY_TEST_WIDTH", test_start
        )
        test_source = source[test_start:test_end]
        self.assertLess(
            test_source.index("fd = mkstemp(path);"),
            test_source.index(
                "ds4_gpu_dspark_shared_test_model(&context.shared_model)"
            ),
        )
        self.assertLess(
            test_source.index("ds4_gpu_tensor_free(context.hc_post);"),
            test_source.index(
                "ds4_gpu_dspark_shared_test_release_model_map()"
            ),
        )
        release_start = source.index(
            "static int ds4_gpu_dspark_shared_test_release_model_map(void)"
        )
        release_end = source.index(
            "/* Port only the fixture producer", release_start
        )
        release_source = source[release_start:release_end]
        for operation in (
            "ds4_gpu_synchronize()",
            "ds4_gpu_model_residency_clear()",
            "ds4_gpu_model_views_clear()",
            "munmap(",
            "g_dspark_shared_test_model_map = NULL",
        ):
            self.assertIn(operation, release_source)

        self.assertEqual(
            uint32_values("g_dspark_routed_test_sum_control").tolist(),
            [
                0x36950000,
                0x2A330000,
                0x44490000,
                0x475F0000,
                0xC1120000,
                0x4E2E0000,
            ],
        )

    def assert_max_abs_drift(
        self,
        label: str,
        actual: np.ndarray,
        expected: np.ndarray,
        limit: float,
    ) -> None:
        self.assertEqual(actual.shape, expected.shape, label)
        maximum = float(np.max(np.abs(actual - expected), initial=0.0))
        self.assertLessEqual(
            maximum,
            limit,
            f"{label} max absolute drift {maximum:.17g} exceeds "
            f"the float32 Metal limit {limit:.17g}",
        )

    @unittest.skipUnless(mlx_optional.available(), "optional MLX is not installed")
    def test_optional_mlx_primitive_parity(self) -> None:
        mlx_version, metal_version, device = mlx_optional.require_pinned_metal()
        self.assertEqual(mlx_version, "0.32.0")
        self.assertEqual(metal_version, "0.32.0")
        self.assertEqual(device, "Device(gpu, 0)")
        markov = self.fixture["cases"]["markovGreedy"]
        numpy_bias = (
            np.asarray(markov["embedding"], dtype=np.float64)[[2, 0, 0]]
            @ np.asarray(markov["projection"], dtype=np.float64).T
        )
        mlx_bias = mlx_optional.markov_step_bias(
            [2, 0, 0],
            np.asarray(markov["embedding"], dtype=np.float64),
            np.asarray(markov["projection"], dtype=np.float64),
        )
        self.assert_max_abs_drift(
            "MLX float32 Markov matmul",
            mlx_bias,
            numpy_bias,
            mlx_optional.MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT,
        )

        confidence = self.fixture["cases"]["confidencePrefix"]
        numpy_result = conditional_confidence(
            np.asarray(confidence["hidden"], dtype=np.float64),
            confidence["previousTokens"],
            np.asarray(confidence["embedding"], dtype=np.float64),
            confidence["projection"],
            threshold=confidence["threshold"],
        )
        mlx_probabilities = mlx_optional.confidence_schedule(
            np.asarray(confidence["hidden"], dtype=np.float64),
            confidence["previousTokens"],
            np.asarray(confidence["embedding"], dtype=np.float64),
            confidence["projection"],
        )
        self.assert_max_abs_drift(
            "MLX float32 confidence projection plus sigmoid",
            mlx_probabilities,
            numpy_result.probabilities,
            mlx_optional.MLX_F32_CONFIDENCE_MAX_ABS_DRIFT,
        )

        capture = self.fixture["cases"]["postLayerHCMean"]
        hidden = self._hc_capture_input(capture)
        numpy_capture = post_layer_hc_mean(hidden)
        mlx_capture = mlx_optional.post_layer_hc_mean(hidden)
        self.assert_max_abs_drift(
            "MLX float32 post-layer HC mean",
            mlx_capture,
            numpy_capture,
            mlx_optional.MLX_F32_HC_MEAN_MAX_ABS_DRIFT,
        )
        with self.assertRaisesRegex(ValueError, r"\[token, 4, 4096\]"):
            mlx_optional.post_layer_hc_mean(hidden[:, :3, :])

        target_capture = self.fixture["cases"]["targetCaptureRows"]
        for phase, token_count_key in (
            ("decode", "decodeTokenCount"),
            ("prefill", "prefillTokenCount"),
        ):
            target_hidden = self._target_capture_input(
                target_capture, target_capture[token_count_key]
            )
            numpy_rows = capture_target_hidden_rows(
                tuple(target_hidden),
                target_capture["layerIds"],
                phase=phase,
            )
            mlx_rows, mlx_token_index, mlx_history = \
                mlx_optional.capture_target_hidden_rows(
                    tuple(target_hidden),
                    target_capture["layerIds"],
                    phase=phase,
                )
            self.assertEqual(mlx_token_index, numpy_rows.token_index)
            self.assert_max_abs_drift(
                f"MLX float32 {phase} target capture rows",
                mlx_rows,
                numpy_rows.rows,
                mlx_optional.MLX_F32_HC_MEAN_MAX_ABS_DRIFT,
            )
            self.assert_max_abs_drift(
                f"MLX float32 {phase} retained target history",
                mlx_history,
                numpy_rows.history_rows,
                mlx_optional.MLX_F32_HC_MEAN_MAX_ABS_DRIFT,
            )

        setup = self.fixture["cases"]["stageSetup"]
        numpy_main = main_project_and_norm(
            np.asarray(setup["targetHidden"], dtype=np.float64),
            np.asarray(setup["mainProjection"], dtype=np.float64),
            setup["mainNormWeight"],
        )
        mlx_main = mlx_optional.main_project_and_norm(
            np.asarray(setup["targetHidden"], dtype=np.float64),
            np.asarray(setup["mainProjection"], dtype=np.float64),
            setup["mainNormWeight"],
        )
        self.assert_max_abs_drift(
            "MLX float32 main projection plus RMSNorm",
            mlx_main,
            numpy_main,
            mlx_optional.MLX_F32_MAIN_PROJECTION_MAX_ABS_DRIFT,
        )

        raw = self.fixture["cases"]["rawContextFinalizer"]
        raw_target = np.asarray(raw["targetHidden"], dtype=np.float64)
        raw_main_projection = np.asarray(
            raw["mainProjection"], dtype=np.float64
        )
        numpy_raw_main = main_project_and_norm(
            raw_target,
            raw_main_projection,
            raw["mainNormWeight"],
            eps=raw["normEps"],
        )
        mlx_raw_main = mlx_optional.main_project_and_norm(
            raw_target,
            raw_main_projection,
            raw["mainNormWeight"],
            eps=raw["normEps"],
        )
        self.assert_max_abs_drift(
            "MLX float32 six-row capture pack/main projection/RMSNorm",
            mlx_raw_main,
            numpy_raw_main,
            mlx_optional.MLX_F32_RAW_CONTEXT_MAIN_MAX_ABS_DRIFT,
        )
        raw_context_projection = self._fixture_matrix(
            raw["contextProjectionGenerator"]
        )
        raw_context_norm = self._fixture_matrix(
            raw["contextNormWeightGenerator"]
        )
        numpy_raw_kv = direct_stage_context_kv(
            numpy_raw_main,
            raw_context_projection,
            raw_context_norm,
            raw["absolutePositions"],
            eps=raw["normEps"],
            rope_theta=raw["ropeTheta"],
        )
        mlx_raw_kv = mlx_optional.direct_stage_context_kv(
            mlx_raw_main,
            raw_context_projection,
            raw_context_norm,
            raw["absolutePositions"],
            eps=raw["normEps"],
            rope_theta=raw["ropeTheta"],
        )
        self.assertEqual(
            mlx_raw_kv.absolute_positions.tolist(), raw["absolutePositions"]
        )
        for label, actual, expected_boundary, limit in (
            (
                "post-Wkv BF16",
                mlx_raw_kv.projected,
                numpy_raw_kv.projected,
                mlx_optional.MLX_BF16_CONTEXT_PROJECTED_MAX_ABS_DRIFT,
            ),
            (
                "post-RMSNorm BF16",
                mlx_raw_kv.normalized,
                numpy_raw_kv.normalized,
                mlx_optional.MLX_BF16_CONTEXT_NORMALIZED_MAX_ABS_DRIFT,
            ),
            (
                "post-RoPE-tail64 BF16",
                mlx_raw_kv.roped,
                numpy_raw_kv.roped,
                mlx_optional.MLX_BF16_CONTEXT_ROPE_MAX_ABS_DRIFT,
            ),
            (
                "post-E4M3FN-dequant BF16 store",
                mlx_raw_kv.stored,
                numpy_raw_kv.stored,
                mlx_optional.MLX_BF16_CONTEXT_STORED_MAX_ABS_DRIFT,
            ),
            (
                "UE8M0 seven-group scales",
                mlx_raw_kv.nonrope_scales,
                numpy_raw_kv.nonrope_scales,
                mlx_optional.MLX_F32_CONTEXT_SCALE_MAX_ABS_DRIFT,
            ),
        ):
            self.assert_max_abs_drift(
                f"MLX raw-context {label}",
                actual,
                expected_boundary,
                limit,
            )

        # Exercise the complete MLX pack-to-store pipeline at every admitted
        # row count.  The full C=6 case above includes the verifier-only row;
        # none of these shapes changes the five-row candidate block contract.
        for rows in range(1, 6):
            mlx_prefix_main = mlx_optional.main_project_and_norm(
                raw_target[:rows],
                raw_main_projection,
                raw["mainNormWeight"],
                eps=raw["normEps"],
            )
            self.assert_max_abs_drift(
                f"MLX raw-context C={rows} main projection/RMSNorm",
                mlx_prefix_main,
                numpy_raw_main[:rows],
                mlx_optional.MLX_F32_RAW_CONTEXT_MAIN_MAX_ABS_DRIFT,
            )
            mlx_prefix_kv = mlx_optional.direct_stage_context_kv(
                mlx_prefix_main,
                raw_context_projection,
                raw_context_norm,
                raw["absolutePositions"][:rows],
                eps=raw["normEps"],
                rope_theta=raw["ropeTheta"],
            )
            self.assertEqual(mlx_prefix_kv.projected.shape, (rows, 512))
            for field, limit in (
                ("projected",
                 mlx_optional.MLX_BF16_CONTEXT_PROJECTED_MAX_ABS_DRIFT),
                ("normalized",
                 mlx_optional.MLX_BF16_CONTEXT_NORMALIZED_MAX_ABS_DRIFT),
                ("roped", mlx_optional.MLX_BF16_CONTEXT_ROPE_MAX_ABS_DRIFT),
                ("stored",
                 mlx_optional.MLX_BF16_CONTEXT_STORED_MAX_ABS_DRIFT),
                ("nonrope_scales",
                 mlx_optional.MLX_F32_CONTEXT_SCALE_MAX_ABS_DRIFT),
            ):
                self.assert_max_abs_drift(
                    f"MLX raw-context C={rows} {field}",
                    getattr(mlx_prefix_kv, field),
                    getattr(numpy_raw_kv, field)[:rows],
                    limit,
                )

        hc = self.fixture["cases"]["hyperConnection"]
        hc_hidden = np.asarray(hc["hidden"], dtype=np.float64)
        hc_function = self._fixture_matrix(hc["function"])
        numpy_reduced, numpy_split = hc_pre(
            hc_hidden,
            hc_function,
            hc["scale"],
            hc["base"],
            iterations=hc["iterations"],
        )
        mlx_reduced, mlx_pre, mlx_post_weights, mlx_combination = \
            mlx_optional.hc_pre(
                hc_hidden,
                hc_function,
                hc["scale"],
                hc["base"],
                iterations=hc["iterations"],
            )
        for label, actual, expected in (
            ("HC pre weights", mlx_pre, numpy_split.pre),
            ("HC post weights", mlx_post_weights, numpy_split.post),
            ("HC Sinkhorn", mlx_combination, numpy_split.combination),
        ):
            self.assert_max_abs_drift(
                f"MLX float32 {label}",
                actual,
                expected,
                mlx_optional.MLX_F32_HC_SPLIT_MAX_ABS_DRIFT,
            )
        self.assert_max_abs_drift(
            "MLX float32 HC pre reduction",
            mlx_reduced,
            numpy_reduced,
            mlx_optional.MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT,
        )
        numpy_expanded = hc_post(hc["branchOutput"], hc_hidden, numpy_split)
        mlx_expanded = mlx_optional.hc_post(
            np.asarray(hc["branchOutput"], dtype=np.float64),
            hc_hidden,
            mlx_post_weights,
            mlx_combination,
        )
        self.assert_max_abs_drift(
            "MLX float32 HC post expansion",
            mlx_expanded,
            numpy_expanded,
            mlx_optional.MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT,
        )
        head_function = self._fixture_matrix(hc["headFunction"])
        numpy_head = hc_head(
            hc_hidden,
            head_function,
            hc["headScale"],
            hc["headBase"],
        )
        mlx_head = mlx_optional.hc_head(
            hc_hidden,
            head_function,
            hc["headScale"],
            hc["headBase"],
        )
        self.assert_max_abs_drift(
            "MLX float32 final HC head",
            mlx_head,
            numpy_head,
            mlx_optional.MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT,
        )

        attention_state, attention_q, attention_draft, attention_sinks = \
            self._dspark_attention_inputs()
        numpy_attention = dspark_attention_official(
            attention_q,
            attention_state,
            attention_draft,
            attention_sinks,
            stage=1,
        ).output
        mlx_attention = mlx_optional.dspark_attention_official(
            attention_q,
            attention_state.rows[1],
            attention_state.length,
            attention_draft[1],
            attention_sinks,
        )
        self.assert_max_abs_drift(
            "MLX BF16 physical-ring online DSpark attention",
            mlx_attention,
            numpy_attention,
            mlx_optional.MLX_BF16_ATTENTION_OUTPUT_MAX_ABS_DRIFT,
        )

        # The compact stage-zero fixture keeps the exact final attention
        # geometry while reducing only hidden/Q-LoRA/output-LoRA fixture
        # widths.  Deterministic compact projections and the conditioned V
        # path keep upstream drift from becoming a whole-chain cascade.
        for committed_count in (2, 128):
            stage_inputs = self._stage_zero_attention_half_inputs(
                committed_count
            )
            numpy_stage = stage_zero_attention_half(**stage_inputs)
            mlx_stage = self._mlx_stage_zero_attention_half(stage_inputs)
            np.testing.assert_array_equal(
                mlx_stage.absolute_positions,
                numpy_stage.absolute_positions,
            )
            for field in numpy_stage.__dataclass_fields__:
                if field in (
                    "absolute_positions",
                    "attention_output",
                    "attention_inverse_roped",
                ):
                    continue
                self.assert_max_abs_drift(
                    f"MLX stage-zero C={committed_count} {field}",
                    getattr(mlx_stage, field),
                    getattr(numpy_stage, field),
                    0.0,
                )
            for field in (
                "attention_output", "attention_inverse_roped"
            ):
                actual_boundary = getattr(mlx_stage, field)
                expected_boundary = getattr(numpy_stage, field)
                self.assert_max_abs_drift(
                    f"MLX stage-zero C={committed_count} {field}",
                    actual_boundary,
                    expected_boundary,
                    mlx_optional.MLX_BF16_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT,
                )
                difference = actual_boundary != expected_boundary
                expected_differences = (
                    0 if committed_count == 2 else
                    2 if field == "attention_output" else 10
                )
                self.assertEqual(
                    int(np.count_nonzero(difference)),
                    expected_differences,
                    f"MLX stage-zero C={committed_count} {field} lane count",
                )
                if expected_differences:
                    actual_codes = np.asarray(
                        actual_boundary, dtype=np.float32
                    ).view(np.uint32) >> np.uint32(16)
                    expected_codes = np.asarray(
                        expected_boundary, dtype=np.float32
                    ).view(np.uint32) >> np.uint32(16)
                    code_distance = np.abs(
                        actual_codes.astype(np.int32)
                        - expected_codes.astype(np.int32)
                    )
                    np.testing.assert_array_equal(
                        code_distance[difference],
                        np.ones(expected_differences, dtype=np.int32),
                    )

        # The payload-first 32/32/4 fixture is the handoff contract for the
        # native white-box Metal test.  C=2 is exact; wrapped C=128 differs
        # only at the online attention result and inverse RoPE publication.
        for committed_count in (2, 128):
            physical = build_physical_stage_zero_fixture(committed_count)
            stage_inputs = dict(physical.inputs)
            numpy_stage = stage_zero_attention_half(**stage_inputs)
            mlx_stage = self._mlx_stage_zero_attention_half(stage_inputs)
            for field in numpy_stage.__dataclass_fields__:
                actual_boundary = getattr(mlx_stage, field)
                expected_boundary = getattr(numpy_stage, field)
                if field in (
                    "attention_output", "attention_inverse_roped"
                ):
                    self.assert_max_abs_drift(
                        f"MLX physical stage-zero C={committed_count} {field}",
                        actual_boundary,
                        expected_boundary,
                        mlx_optional.
                        MLX_BF16_PHYSICAL_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT,
                    )
                    difference = actual_boundary != expected_boundary
                    expected_differences = 0 if committed_count == 2 else 6
                    self.assertEqual(
                        int(np.count_nonzero(difference)), expected_differences
                    )
                    if expected_differences:
                        actual_codes = np.asarray(
                            actual_boundary, dtype=np.float32
                        ).view(np.uint32) >> np.uint32(16)
                        expected_codes = np.asarray(
                            expected_boundary, dtype=np.float32
                        ).view(np.uint32) >> np.uint32(16)
                        np.testing.assert_array_equal(
                            np.abs(
                                actual_codes.astype(np.int32)
                                - expected_codes.astype(np.int32)
                            )[difference],
                            np.ones(expected_differences, dtype=np.int32),
                        )
                else:
                    self.assert_max_abs_drift(
                        f"MLX physical stage-zero C={committed_count} {field}",
                        actual_boundary,
                        expected_boundary,
                        0.0,
                    )

        ffn_fixture = build_physical_ffn_fixture()
        numpy_ffn = dspark_reference.stage_ffn_moe_payload_first(
            **ffn_fixture.inputs
        )
        mlx_ffn = mlx_optional.stage_ffn_moe_payload_first(
            **ffn_fixture.inputs
        )
        np.testing.assert_array_equal(
            mlx_ffn.selected_experts, numpy_ffn.selected_experts
        )
        self.assertEqual(
            set(mlx_optional.MLX_FFN_OPERATION_MAX_ABS_DRIFT),
            set(numpy_ffn.__dataclass_fields__) - {"selected_experts"},
        )
        for field, limit in (
            mlx_optional.MLX_FFN_OPERATION_MAX_ABS_DRIFT.items()
        ):
            self.assert_max_abs_drift(
                f"MLX payload-first FFN {field}",
                getattr(mlx_ffn, field),
                getattr(numpy_ffn, field),
                limit,
            )

    def test_runtime_capture_call_sites_match_oracle_contract(self) -> None:
        source = DS4_SOURCE_PATH.read_text(encoding="utf-8")
        metal_source = DS4_METAL_SOURCE_PATH.read_text(encoding="utf-8")

        def function_tokens(name: str) -> list[str]:
            return _c_tokens(_c_function_body(source, name))

        def metal_function_tokens(name: str) -> list[str]:
            return _c_tokens(_c_function_body(metal_source, name))

        def require_sequence(
                tokens: list[str], sequence: list[str], message: str,
                *, start: int = 0, end: int | None = None) -> int:
            index = _token_sequence_index(
                tokens, sequence, start=start, end=end)
            self.assertNotEqual(index, -1, message)
            return index

        def capture_calls(
                tokens: list[str], expected_count: int,
        ) -> list[tuple[int, int, list[list[str]]]]:
            calls = _c_calls(tokens, "metal_graph_dspark_capture_after_layer")
            self.assertEqual(len(calls), expected_count)
            return calls

        def assert_capture_arguments(
                calls: list[tuple[int, int, list[list[str]]]],
                expected: list[list[str]]) -> None:
            for _, _, arguments in calls:
                self.assertEqual(arguments, expected)

        def assert_swap_between(
                tokens: list[str], producer_end: int, consumer_start: int,
                current: str, next_buffer: str) -> None:
            swap = [
                "g", "->", current, "=", "g", "->", next_buffer, ";",
                "g", "->", next_buffer, "=", "tmp", ";",
            ]
            require_sequence(
                tokens,
                swap,
                f"{current}/{next_buffer} must swap after the layer producer "
                "and before DSpark capture",
                start=producer_end + 1,
                end=consumer_start,
            )

        # The array order is the layer-to-slot mapping consumed by the oracle;
        # accepting a permutation would silently concatenate the wrong rows.
        graph_source = DSPARK_GRAPH_PATH.read_text(encoding="utf-8")
        contract_tokens = _c_tokens(_strip_c_noncode(graph_source))
        require_sequence(
            contract_tokens,
            [".", "target_layer", "=", "{", "40", ",", "41", ",", "42", "}"],
            "runtime target layers must retain oracle slot order 40/41/42",
        )
        slot_tokens = _c_tokens(_c_function_body(
            graph_source, "ds4_dspark_capture_slot_after_layer"))
        require_sequence(
            slot_tokens,
            [
                "DS4_DSPARK_CAPTURE_CONTRACT", ".", "target_layer", "[",
                "index", "]", "==", "layer",
            ],
            "slot lookup must compare the indexed target-layer contract",
        )
        require_sequence(
            slot_tokens,
            ["if", "(", "slot", ")", "*", "slot", "=", "index", ";"],
            "the matched contract index must remain the capture slot",
        )

        decode_arguments = [
            ["g"], ["il"], ["g", "->", "cur_hc"], ["1u"], ["0u"],
        ]
        resident = function_tokens("metal_graph_encode_token_raw_swa")
        resident_captures = capture_calls(resident, 1)
        assert_capture_arguments(resident_captures, decode_arguments)
        resident_producers = _c_calls(
            resident, "metal_graph_encode_decode_layer")
        self.assertEqual(len(resident_producers), 1)
        assert_swap_between(
            resident,
            resident_producers[0][1],
            resident_captures[0][0],
            "cur_hc",
            "after_ffn_hc",
        )

        streaming = function_tokens("metal_graph_eval_token_raw_swa_streaming")
        streaming_captures = capture_calls(streaming, 2)
        assert_capture_arguments(streaming_captures, decode_arguments)
        streaming_producers = _c_calls(
            streaming, "metal_graph_encode_decode_layer")
        self.assertEqual(len(streaming_producers), 2)
        for capture, producer in zip(streaming_captures, streaming_producers):
            self.assertLess(producer[1], capture[0])
            assert_swap_between(
                streaming,
                producer[1],
                capture[0],
                "cur_hc",
                "after_ffn_hc",
            )

        batch_helper = function_tokens("metal_graph_encode_layer_batch")
        batch_attention = _c_calls(
            batch_helper, "metal_graph_encode_layer_attention_batch")
        batch_ffn = _c_calls(batch_helper, "metal_graph_encode_layer_ffn_batch")
        self.assertEqual(len(batch_attention), 1)
        self.assertEqual(len(batch_ffn), 1)
        self.assertLess(batch_attention[0][1], batch_ffn[0][0])
        require_sequence(
            batch_helper,
            [
                "g", "->", "batch_cur_hc", "=", "g", "->",
                "batch_next_hc", ";", "g", "->", "batch_next_hc", "=",
                "tmp", ";",
            ],
            "complete prefill helper must publish post-FFN HC before returning",
            start=batch_ffn[0][1] + 1,
        )

        prefill = function_tokens("metal_graph_prefill_layer_major")
        prefill_captures = capture_calls(prefill, 3)
        prefill_arguments = [
            ["g"],
            ["il"],
            ["g", "->", "batch_cur_hc"],
            ["n_tokens"],
            ["n_tokens", "-", "1u"],
        ]
        assert_capture_arguments(prefill_captures, prefill_arguments)
        producers: list[tuple[int, int, str]] = []
        for producer_name in (
                "metal_graph_encode_layer_batch",
                "metal_graph_encode_layer_ffn_batch"):
            producers.extend(
                (start, end, producer_name)
                for start, end, _ in _c_calls(prefill, producer_name)
            )
        nearest_producers: list[tuple[int, int, str]] = []
        for capture_start, _, _ in prefill_captures:
            preceding = [
                producer for producer in producers
                if producer[1] < capture_start
            ]
            self.assertTrue(preceding, "capture must follow a layer producer")
            nearest_producers.append(max(preceding, key=lambda item: item[0]))
        self.assertEqual(
            [producer[2] for producer in nearest_producers],
            [
                "metal_graph_encode_layer_batch",
                "metal_graph_encode_layer_ffn_batch",
                "metal_graph_encode_layer_batch",
            ],
        )
        # The profiled split branch calls the FFN half directly, so its swap
        # cannot rely on the complete-layer helper checked above.
        assert_swap_between(
            prefill,
            nearest_producers[1][1],
            prefill_captures[1][0],
            "batch_cur_hc",
            "batch_next_hc",
        )

        suffix_verifier = function_tokens("metal_graph_verify_suffix_tops")
        exact_verifier = function_tokens("metal_graph_verify_decode2_exact")
        for verifier_name, verifier in (
                ("metal_graph_verify_suffix_tops", suffix_verifier),
                ("metal_graph_verify_decode2_exact", exact_verifier)):
            self.assertEqual(
                _c_calls(verifier, "metal_graph_dspark_capture_after_layer"),
                [],
                f"{verifier_name} must not publish through the current capture tap",
            )

        suffix_begin = _c_calls(
            suffix_verifier, "metal_graph_dspark_verify_capture_begin")
        suffix_scratch = _c_calls(
            suffix_verifier,
            "metal_graph_dspark_verify_capture_after_layer_batch")
        suffix_layers = _c_calls(
            suffix_verifier, "metal_graph_encode_layer_batch")
        self.assertEqual(len(suffix_begin), 1)
        self.assertEqual(
            suffix_begin[0][2], [["g"], ["start"], ["n_tokens"]])
        self.assertEqual(len(suffix_layers), 1)
        self.assertEqual(len(suffix_scratch), 1)
        self.assertEqual(
            suffix_scratch[0][2],
            [
                ["g"], ["il"], ["g", "->", "batch_cur_hc"],
                ["n_tokens"],
            ],
        )
        self.assertLess(suffix_begin[0][1], suffix_layers[0][0])
        self.assertLess(suffix_layers[0][1], suffix_scratch[0][0])
        suffix_invalidations = _c_calls(
            suffix_verifier, "metal_graph_dspark_capture_invalidate")
        self.assertGreaterEqual(
            len(suffix_invalidations), 2,
            "suffix-verifier validation and execution failures must discard "
            "scratch capture",
        )
        require_sequence(
            suffix_verifier,
            [
                "if", "(", "!", "metal_graph_dspark_verify_capture_begin",
                "(", "g", ",", "start", ",", "n_tokens", ")", ")",
                "{", "metal_graph_dspark_capture_invalidate", "(", "g",
                ")", ";", "return", "false", ";", "}",
            ],
            "a failed suffix capture begin must invalidate before returning",
        )
        require_sequence(
            suffix_verifier,
            [
                "if", "(", "!", "ok", ")",
                "metal_graph_dspark_capture_invalidate", "(", "g", ")",
                ";", "return", "ok", ";",
            ],
            "suffix completion failure must invalidate before returning",
        )

        exact_begin = _c_calls(
            exact_verifier, "metal_graph_dspark_verify_capture_begin")
        exact_scratch = _c_calls(
            exact_verifier,
            "metal_graph_dspark_verify_capture_after_layer_batch")
        exact_layers = _c_calls(
            exact_verifier, "metal_graph_encode_decode_layer")
        self.assertEqual(len(exact_begin), 1)
        self.assertEqual(exact_begin[0][2], [["g"], ["start"], ["2u"]])
        self.assertEqual(len(exact_layers), 2)
        self.assertEqual(len(exact_scratch), 1)
        self.assertEqual(
            exact_scratch[0][2],
            [["g"], ["il"], ["next_pair_hc"], ["2u"]],
        )
        self.assertLess(exact_begin[0][1], exact_layers[0][0])
        self.assertLess(exact_layers[0][1], exact_layers[1][0])
        self.assertLess(exact_layers[1][1], exact_scratch[0][0])
        exact_invalidations = _c_calls(
            exact_verifier, "metal_graph_dspark_capture_invalidate")
        self.assertGreaterEqual(
            len(exact_invalidations), 1,
            "exact-verifier failure must discard scratch capture",
        )
        require_sequence(
            exact_verifier,
            [
                "if", "(", "!", "ok", ")",
                "metal_graph_dspark_capture_invalidate", "(", "g", ")",
                ";",
            ],
            "exact-verifier completion failure must invalidate capture",
        )

        verify_begin = function_tokens("metal_graph_dspark_verify_capture_begin")
        require_sequence(
            verify_begin,
            [
                "const", "uint64_t", "end", "=", "start", "+", "n_rows",
                ";",
            ],
            "verifier capture generation must target the full speculative end",
        )
        begin_state = _c_calls(
            verify_begin, "ds4_dspark_capture_state_begin")
        self.assertEqual(len(begin_state), 1)
        self.assertEqual(
            begin_state[0][2],
            [["&", "g", "->", "dspark_verify_capture_state"], ["end"]],
        )
        for assignment in (
                ["g", "->", "dspark_verify_capture_active", "=", "true", ";"],
                ["g", "->", "dspark_verify_capture_rows", "=", "n_rows", ";"],
                ["g", "->", "dspark_verify_capture_start", "=", "start", ";"]):
            require_sequence(
                verify_begin, assignment,
                "verifier begin must retain its isolated scratch transaction")

        for scratch_helper in (
                "metal_graph_dspark_verify_capture_after_layer_batch",):
            scratch_tokens = function_tokens(scratch_helper)
            require_sequence(
                scratch_tokens,
                ["g", "->", "dspark_verify_capture", "[", "slot", "]"],
                f"{scratch_helper} must write verifier scratch storage",
            )
            self.assertEqual(
                _token_sequence_index(
                    scratch_tokens,
                    ["g", "->", "dspark_capture", "[", "slot", "]"]),
                -1,
                f"{scratch_helper} must not overwrite consumable capture rows",
            )

        publish = function_tokens("metal_graph_dspark_verify_capture_publish")
        device_publish = metal_function_tokens(
            "ds4_gpu_dspark_publish_history_tensor")
        publish_copies = _c_calls(device_publish, "ds4_gpu_tensor_copy")
        self.assertEqual(
            len(publish_copies), 3,
            "shared publication helper needs first history span, optional wrap span, and "
            "one current-frontier copy",
        )
        self.assertEqual(
            publish_copies[2][2],
            [
                ["current"],
                ["0"],
                ["candidate"],
                [
                    "(", "uint64_t", ")", "(", "committed_rows", "-",
                    "1u", ")", "*", "row_bytes",
                ],
                ["row_bytes"],
            ],
        )
        host_publish_calls = _c_calls(
            publish, "ds4_gpu_dspark_publish_history_tensor")
        self.assertEqual(len(host_publish_calls), 1)
        self.assertEqual(
            host_publish_calls[0][2],
            [
                ["g", "->", "dspark_capture", "[", "stage", "]"],
                ["g", "->", "dspark_history", "[", "stage", "]"],
                ["g", "->", "dspark_verify_capture", "[", "stage", "]"],
                ["committed_rows"],
                ["plan", ".", "first_physical_row"],
                ["plan", ".", "first_rows"],
                ["plan", ".", "second_rows"],
            ],
            "host publication must delegate exact geometry to the shared device helper",
        )
        self.assertEqual(
            len(_c_calls(publish, "ds4_gpu_tensor_copy")), 0,
            "host publication must not retain a mirrored device implementation",
        )
        require_sequence(
            publish,
            [
                "ds4_dspark_history_state_begin", "(", "&", "g", "->",
                "dspark_history_state", ",", "g", "->",
                "dspark_verify_capture_start", ",", "committed_rows", ")",
            ],
            "publication must begin one transaction for all committed rows",
        )
        ready_checks = _c_calls(
            publish, "ds4_dspark_capture_state_ready_at")
        self.assertEqual(len(ready_checks), 1)
        self.assertEqual(
            ready_checks[0][2],
            [["&", "g", "->", "dspark_capture_state"], ["frontier"]],
        )
        history_ready_checks = _c_calls(
            publish, "ds4_dspark_history_state_current_ready_at")
        self.assertEqual(len(history_ready_checks), 1)
        self.assertEqual(
            history_ready_checks[0][2],
            [["&", "g", "->", "dspark_history_state"], ["frontier"]],
        )
        publish_aborts = _c_calls(
            publish, "metal_graph_dspark_verify_capture_abort")
        self.assertEqual(len(publish_aborts), 1)
        current_finish = _c_calls(
            publish, "ds4_dspark_capture_state_finish")
        history_finish = _c_calls(
            publish, "ds4_dspark_history_state_finish")
        self.assertEqual(len(current_finish), 1)
        self.assertEqual(len(history_finish), 1)
        self.assertLess(current_finish[0][1], publish_aborts[0][0])
        self.assertLess(history_finish[0][1], publish_aborts[0][0])
        self.assertGreaterEqual(
            len(_c_calls(publish, "metal_graph_dspark_capture_invalidate")),
            3,
            "validation, publication, and readiness failures must invalidate "
            "current and history state",
        )

        speculative = function_tokens("ds4_session_eval_speculative_argmax")
        publish_calls = _c_calls(
            speculative, "metal_graph_dspark_verify_capture_publish")
        self.assertEqual(
            [call[2] for call in publish_calls],
            [
                [
                    ["&", "s", "->", "graph"], ["2u"],
                    ["(", "uint64_t", ")", "start", "+", "2u"],
                ],
                [
                    ["&", "s", "->", "graph"], ["1u"],
                    ["(", "uint64_t", ")", "start", "+", "1u"],
                ],
                [
                    ["&", "s", "->", "graph"],
                    ["(", "uint32_t", ")", "draft_n"],
                    [
                        "(", "uint64_t", ")", "start", "+", "(",
                        "uint32_t", ")", "draft_n",
                    ],
                ],
                [
                    ["&", "s", "->", "graph"], ["1u"],
                    ["(", "uint64_t", ")", "start", "+", "1u"],
                ],
                [
                    ["&", "s", "->", "graph"],
                    ["(", "uint32_t", ")", "commit_drafts"],
                    [
                        "(", "uint64_t", ")", "start", "+", "(",
                        "uint32_t", ")", "commit_drafts",
                    ],
                ],
            ],
            "each acceptance path must publish its committed row count/frontier",
        )
        invalidations = _c_calls(
            speculative, "metal_graph_dspark_capture_invalidate")
        verifier_aborts = _c_calls(
            speculative, "metal_graph_dspark_verify_capture_abort")
        restores = _c_calls(speculative, "spec_frontier_restore")
        self.assertGreaterEqual(len(invalidations), 1)
        self.assertGreaterEqual(len(verifier_aborts), 1)
        self.assertGreaterEqual(len(restores), 1)
        cleanups = sorted(
            invalidations + verifier_aborts, key=lambda call: call[0]
        )
        previous_restore_end = -1
        for restore_start, restore_end, _ in restores:
            cleanups_before_restore = [
                cleanup for cleanup in cleanups
                if previous_restore_end < cleanup[1] < restore_start
            ]
            self.assertTrue(
                cleanups_before_restore,
                "every speculative frontier restore must first abort verifier "
                "scratch or invalidate all capture/history state",
            )
            previous_restore_end = restore_end
        replay_evaluations = [
            evaluation for evaluation in _c_calls(
                speculative, "metal_graph_eval_token_raw_swa")
            if any(restore[1] < evaluation[0] for restore in restores)
        ]
        self.assertEqual(
            len(replay_evaluations),
            2,
            "both exact replay lanes must remain classified by this test",
        )
        for replay in replay_evaluations:
            preceding_restore = max(
                (restore for restore in restores if restore[1] < replay[0]),
                key=lambda call: call[0],
            )
            self.assertTrue(any(
                invalidation[1] < preceding_restore[0]
                for invalidation in invalidations
            ))

        decode_prefill = function_tokens(
            "metal_graph_prefill_decode_streaming_range")
        invalidations = _c_calls(
            decode_prefill, "metal_graph_dspark_capture_invalidate")
        evaluations = _c_calls(
            decode_prefill, "metal_graph_eval_token_raw_swa")
        self.assertGreaterEqual(
            len(invalidations), 1,
            "decode-style prefill cancellation/failure must invalidate history",
        )
        self.assertEqual(len(evaluations), 1)
        loop = require_sequence(
            decode_prefill, ["for", "("],
            "decode-style prefill must retain its token loop")
        last = require_sequence(
            decode_prefill,
            [
                "const", "bool", "last", "=", "i", "+", "1u", "==",
                "n_tokens", ";",
            ],
            "the final decode-style prefill token still owns logits output",
            start=loop,
        )
        completed = require_sequence(
            decode_prefill,
            [
                "completed_tokens", "++", ";",
            ],
            "each successful decode-style prefill row must become committed",
            start=evaluations[0][1] + 1,
        )
        self.assertEqual(
            _token_sequence_index(
                decode_prefill, ["dspark_capture_suspended"]),
            -1,
            "official prompt history must capture every decode-style row, not "
            "suspend all but the frontier",
        )
        self.assertLess(loop, last)
        self.assertLess(last, evaluations[0][0])
        self.assertLess(evaluations[0][1], completed)

        allocator = function_tokens("metal_graph_alloc_raw_cap")
        state_initializers = _c_calls(
            allocator, "ds4_dspark_capture_state_init")
        self.assertEqual(
            [call[2] for call in state_initializers],
            [
                [["&", "g", "->", "dspark_capture_state"],
                 ["enable_dspark"]],
                [["&", "g", "->", "dspark_verify_capture_state"],
                 ["enable_dspark"]],
            ],
        )
        history_initializers = _c_calls(
            allocator, "ds4_dspark_history_state_init")
        self.assertEqual(len(history_initializers), 1)
        self.assertEqual(
            history_initializers[0][2],
            [["&", "g", "->", "dspark_history_state"], ["enable_dspark"]],
        )
        enabled_block = require_sequence(
            allocator,
            ["if", "(", "enable_dspark", ")", "{"],
            "capture tensors must have an explicit enable_dspark allocation gate",
        )
        block_open = enabled_block + 4
        block_close = _matching_token(allocator, block_open, "{", "}")
        for capture_field in (
                "dspark_capture", "dspark_history", "dspark_verify_capture"):
            capture_allocation = [
                "g", "->", capture_field, "[", "stage", "]", "=",
                "ds4_gpu_tensor_alloc", "(",
            ]
            allocation_index = require_sequence(
                allocator,
                capture_allocation,
                f"enabled DSpark graphs must allocate {capture_field} rows",
                start=block_open + 1,
                end=block_close,
            )
            self.assertEqual(
                _token_sequence_index(
                    allocator, capture_allocation, start=allocation_index + 1),
                -1,
                f"{capture_field} storage must not be allocated outside its gate",
            )
        require_sequence(
            allocator,
            [
                "(", "!", "enable_dspark", "||", "(", "g", "->",
                "dspark_capture", "[", "0", "]", "&&", "g", "->",
                "dspark_capture", "[", "1", "]", "&&", "g", "->",
                "dspark_capture", "[", "2", "]", "&&", "g", "->",
                "dspark_history", "[", "0", "]", "&&", "g", "->",
                "dspark_history", "[", "1", "]", "&&", "g", "->",
                "dspark_history", "[", "2", "]", "&&", "g", "->",
                "dspark_verify_capture", "[", "0", "]", "&&", "g", "->",
                "dspark_verify_capture", "[", "1", "]", "&&", "g", "->",
                "dspark_verify_capture", "[", "2", "]", ")", ")",
            ],
            "allocator success must require all rows only when DSpark is enabled",
        )

        allocation_callers = {
            "metal_graph_alloc": ["false"],
            "metal_graph_prompt_logits_test": ["false"],
            "generate_metal_graph_raw_swa": [
                "model", "->", "native_dspark_store_v2"],
            "ds4_engine_collect_imatrix": ["false"],
            "ds4_session_create": [
                "e", "->", "model", ".", "native_dspark_store_v2"],
        }
        for caller, expected_enable_argument in allocation_callers.items():
            calls = _c_calls(
                function_tokens(caller), "metal_graph_alloc_raw_cap")
            self.assertEqual(len(calls), 1, caller)
            self.assertEqual(len(calls[0][2]), 8, caller)
            self.assertEqual(calls[0][2][-1], expected_enable_argument, caller)

        full_source_tokens = _c_tokens(_strip_c_noncode(source))
        runtime_capture_calls = [
            call for call in _c_calls(
                full_source_tokens, "metal_graph_dspark_capture_after_layer")
            if call[1] + 1 >= len(full_source_tokens)
            or full_source_tokens[call[1] + 1] != "{"
        ]
        self.assertEqual(
            len(runtime_capture_calls),
            6,
            "every production capture tap must be classified by this test",
        )
        raw_allocator_calls = [
            call for call in _c_calls(full_source_tokens, "metal_graph_alloc_raw_cap")
            if call[1] + 1 >= len(full_source_tokens)
            or full_source_tokens[call[1] + 1] != "{"
        ]
        self.assertEqual(
            len(raw_allocator_calls),
            len(allocation_callers),
            "every raw graph allocation must declare its DSpark enable policy",
        )

    def test_generated_fixture_is_current(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "tools" / "dspark_oracle" / "generate_fixtures.py"),
            "--check",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("fixture check: OK", completed.stdout)

    def test_fixture_generator_does_not_import_oracle_implementation(self) -> None:
        tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(f"{module}.{alias.name}" for alias in node.names)
        coupled = [
            name
            for name in imports
            if name.startswith("tools.dspark_oracle")
            or name.startswith("reference")
            or ".reference" in name
        ]
        self.assertEqual(coupled, [])


if __name__ == "__main__":
    unittest.main()
