#!/usr/bin/env python3
"""Model-free tests for the development-only DSpark numerical oracle."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.dspark_oracle import (  # noqa: E402
    MetadataError,
    conditional_confidence,
    markov_greedy_draft,
    post_layer_hc_mean,
    speculative_sample_exact,
    validate_0731_metadata,
)
from tools.dspark_oracle import mlx_optional  # noqa: E402


FIXTURE_PATH = Path(__file__).with_name("fixtures-v1.json")
PROVENANCE_PATH = ROOT / "tools" / "dspark_oracle" / "provenance.json"
GENERATOR_PATH = ROOT / "tools" / "dspark_oracle" / "generate_fixtures.py"


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
        self.assertEqual(hf.block_size, 5)
        self.assertEqual(hf.stage_count, 3)
        self.assertEqual(hf, gguf)

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

    def test_pinned_provenance(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schemaVersion"], 2)
        self.assertFalse(provenance["implementation"]["productionDependency"])
        self.assertFalse(provenance["implementation"]["externalRuntimeCodeCopied"])
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
            1.0e-6,
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
