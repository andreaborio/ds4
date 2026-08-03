#!/usr/bin/env python3
"""Generate or byte-check the independently declared DSpark oracle fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tests" / "dspark" / "fixtures-v1.json"


# The generator intentionally imports no oracle implementation.  Inputs and
# expected outputs below are a second specification against which reference.py
# is tested.  The arithmetic notes make the larger fixture reviewable without
# executing the code under test:
#
# * Markov rows are base_logits + embedding[previous] @ projection.T.  Feeding
#   each selected token into the next row yields tokens [0, 0, 2].
# * Confidence logits are the hand-summed dot products 0.625, 0.525 and 0.175.
#   The listed probabilities are sigmoid(logit).  With threshold 0.6, the
#   first two positions pass independently and the third ends the prefix.
#   Multiplying them would stop after one position and is intentionally wrong.
# * Rejection thresholds are the literal p(x)/q(x) ratios [0.8, 0.5, >1].  The
#   second draw rejects and max(p-q, 0) normalizes to [0.8, 0.2, 0, 0].
# * The all-accepted proposal equals the first three target rows, so every
#   threshold is one and categorical u=0.65 selects token 3 from the bonus row.
EMBEDDING = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
MARKOV_PROJECTION = [
    [0.2, -0.1],
    [-0.4, 0.3],
    [0.1, 0.5],
    [-0.2, -0.2],
]
BASE_LOGITS = [
    [1.0, 0.5, -1.0, 0.0],
    [0.2, 0.1, 0.0, -0.1],
    [-0.5, 0.2, 0.4, 0.3],
]
MARKOV_STEP_LOGITS = [
    [1.1, 0.39999999999999997, -0.4, -0.4],
    [0.4, -0.30000000000000004, 0.1, -0.30000000000000004],
    [-0.3, -0.2, 0.5, 0.09999999999999998],
]

CONFIDENCE_HIDDEN = [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]]
CONFIDENCE_PROJECTION = [0.5, -0.25, 0.4, 0.1]
CONFIDENCE_LOGITS = [0.625, 0.5250000000000001, 0.17500000000000004]
CONFIDENCE_PROBABILITIES = [
    0.6513548646660542,
    0.6283161882953663,
    0.5436386872370789,
]

TARGET_PROBABILITIES = [
    [0.1, 0.4, 0.2, 0.3],
    [0.5, 0.1, 0.2, 0.2],
    [0.2, 0.2, 0.5, 0.1],
    [0.1, 0.2, 0.3, 0.4],
]
REJECT_PROPOSAL = [
    [0.2, 0.5, 0.1, 0.2],
    [0.3, 0.05, 0.4, 0.25],
    [0.1, 0.4, 0.4, 0.1],
]
ACCEPT_PROPOSAL = [
    [0.1, 0.4, 0.2, 0.3],
    [0.5, 0.1, 0.2, 0.2],
    [0.2, 0.2, 0.5, 0.1],
]

HC_CAPTURE_FIXTURE = {
    "shape": [3, 4, 4096],
    "generator": {
        "tokenScale": 0.25,
        "hcScale": 1.5,
        "dimensionPeriod": 257,
        "dimensionOffset": -128,
        "dimensionScale": 0.03125,
    },
    "expectedShape": [3, 4096],
    "expectedSamples": [
        {"index": [0, 0], "value": -1.75},
        {"index": [0, 128], "value": 2.25},
        {"index": [0, 256], "value": 6.25},
        {"index": [1, 257], "value": -1.5},
        {"index": [2, 4095], "value": 6.25},
    ],
}


def _declared_metadata() -> dict[str, dict[str, Any]]:
    """Return source-pinned values without loading metadata.py or schema.json."""

    return {
        "hf": {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "vocab_size": 129280,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "dspark_block_size": 5,
            "dspark_markov_rank": 256,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42],
        },
        "gguf": {
            "general.architecture": "deepseek4",
            "deepseek4.block_count": 43,
            "deepseek4.vocab_size": 129280,
            "deepseek4.expert_count": 256,
            "deepseek4.expert_used_count": 6,
            "dspark.block_size": 5,
            "dspark.markov_rank": 256,
            "dspark.noise_token_id": 128799,
            "dspark.target_layer_ids": [40, 41, 42],
            "dspark.stage_count": 3,
            "dspark.n_layers": 3,
        },
    }


def build_fixture() -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "profile": "deepseek-v4-flash-0731-dspark",
        "generatedBy": "tools/dspark_oracle/generate_fixtures.py",
        "numericType": "float64",
        "cases": {
            "postLayerHCMean": HC_CAPTURE_FIXTURE,
            "markovGreedy": {
                "baseLogits": BASE_LOGITS,
                "firstPreviousToken": 2,
                "embedding": EMBEDDING,
                "projection": MARKOV_PROJECTION,
                "expected": {
                    "tokens": [0, 0, 2],
                    "stepLogits": MARKOV_STEP_LOGITS,
                },
            },
            "confidencePrefix": {
                "hidden": CONFIDENCE_HIDDEN,
                "previousTokens": [2, 0, 0],
                "embedding": EMBEDDING,
                "projection": CONFIDENCE_PROJECTION,
                "threshold": 0.6,
                "expected": {
                    "logits": CONFIDENCE_LOGITS,
                    "probabilities": CONFIDENCE_PROBABILITIES,
                    "keep": 2,
                },
            },
            "samplingRejected": {
                "targetProbabilities": TARGET_PROBABILITIES,
                "draftTokens": [1, 2, 2],
                "draftProbabilities": REJECT_PROPOSAL,
                "acceptanceUniforms": [0.1, 0.8, 0.0],
                "categoricalUniforms": [0.0, 0.85, 0.0, 0.0],
                "expected": {
                    "accepted": 1,
                    "acceptanceThresholds": [0.8, 0.5, 1.0],
                    "targetRow": 1,
                    "residualProbabilities": [0.8, 0.2, 0.0, 0.0],
                    "replacementToken": 1,
                    "committedTokens": [1, 1],
                },
            },
            "samplingAllAccepted": {
                "targetProbabilities": TARGET_PROBABILITIES,
                "draftTokens": [1, 0, 2],
                "draftProbabilities": ACCEPT_PROPOSAL,
                "acceptanceUniforms": [0.9, 0.9, 0.9],
                "categoricalUniforms": [0.0, 0.0, 0.0, 0.65],
                "expected": {
                    "accepted": 3,
                    "acceptanceThresholds": [1.0, 1.0, 1.0],
                    "targetRow": 3,
                    "replacementToken": 3,
                    "committedTokens": [1, 0, 2, 3],
                },
            },
            "metadata0731": _declared_metadata(),
        },
    }


def render_fixture() -> str:
    return json.dumps(build_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_fixture()
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"fixture check: FAIL: {exc}", file=sys.stderr)
            return 1
        if current != rendered:
            print(
                f"fixture check: FAIL: regenerate {args.output}",
                file=sys.stderr,
            )
            return 1
        print(f"fixture check: OK: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
