#!/usr/bin/env python3
"""Model-free tests for the development-only DSpark numerical oracle."""

from __future__ import annotations

import ast
import inspect
import json
import os
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
    MetadataError,
    append_raw_cache,
    conditional_confidence,
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
    def _fixture_matrix(description: object) -> np.ndarray:
        if isinstance(description, dict):
            return np.full(
                tuple(int(item) for item in description["shape"]),
                float(description["fill"]),
                dtype=np.float64,
            )
        return np.asarray(description, dtype=np.float64)

    def test_stage_zero_main_projection_norm_and_noise_layout(self) -> None:
        case = self.fixture["cases"]["stageSetup"]
        target = np.asarray(case["targetHidden"], dtype=np.float64)
        projected = main_project_and_norm(
            target,
            np.asarray(case["mainProjection"], dtype=np.float64),
            case["mainNormWeight"],
        )
        setup = prepare_stage_zero(
            target,
            case["acceptedEmbedding"],
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

    def test_main_projection_requires_three_captures(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            main_project_and_norm(
                np.zeros((2, 2)), np.zeros((2, 4)), np.ones(2)
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

        current = self._raw_cache_rows(
            np.asarray([case["proposalPosition"]]), case
        )[0]
        draft = self._raw_cache_drafts(case)
        before = np.array(state128.rows, copy=True)
        view = proposal_raw_cache_view(
            state128, case["proposalPosition"], current, draft
        )
        self.assertEqual(
            view.shape,
            (DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WINDOW + 1 + 5,
             DSPARK_RAW_CACHE_WIDTH),
        )
        self._assert_raw_samples(view, expected["proposalSamples"], "view")
        np.testing.assert_array_equal(state128.rows, before)

        committed = append_raw_cache(
            state128, case["proposalPosition"], current
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

    def test_raw_cache_rejects_non_0731_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity must be 128"):
            empty_raw_cache(127, DSPARK_RAW_CACHE_WIDTH)
        with self.assertRaisesRegex(ValueError, "width must be 512"):
            empty_raw_cache(DSPARK_RAW_CACHE_WINDOW, 511)

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
            mlx_optional.MLX_F32_HC_SPLIT_MAX_ABS_DRIFT,
            1.0e-7,
        )
        self.assertEqual(
            mlx_optional.MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT,
            5.0e-7,
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
