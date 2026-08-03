#!/usr/bin/env python3
"""Model-free tests for the development-only DSpark numerical oracle."""

from __future__ import annotations

import ast
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
    append_raw_cache,
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
    prefill_raw_cache,
    prepare_stage_zero,
    proposal_token_layout,
    proposal_raw_cache_view,
    rms_norm,
    run_synthetic_stage_chain,
    speculative_sample_exact,
    validate_0731_metadata,
)
from tools.dspark_oracle import mlx_optional  # noqa: E402
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
        self.assertGreater(
            mlx_optional.MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT,
            mlx_optional.MLX_F32_CONFIDENCE_MAX_ABS_DRIFT,
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
