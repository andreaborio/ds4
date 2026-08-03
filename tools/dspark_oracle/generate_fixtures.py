#!/usr/bin/env python3
"""Generate or byte-check the independently declared DSpark oracle fixture."""

from __future__ import annotations

import argparse
import json
import math
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
# * Non-degenerate HC and final-head expected arrays below are frozen values
#   independently evaluated from the pinned official equations.  They are not
#   recomputed here, so changing a weight, lane order, normalization, or axis
#   makes the fixture fail instead of silently regenerating a matching answer.
# * Target capture values use layer*65536 + token*8192 + lane*512 plus a
#   periodic dimension term.  Averaging lanes 0/1/2/3 contributes exactly 768;
#   the prefill case has 130 rows so its frontier is distinct from the retained
#   last-128 history starting at local row two.
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

TARGET_CAPTURE_FIXTURE = {
    "layerIds": [40, 41, 42],
    "shapePerLayer": [4, 4096],
    "generator": {
        "layerScale": 65536.0,
        "tokenScale": 8192.0,
        "hcScale": 512.0,
        "dimensionPeriod": 257,
        "dimensionOffset": -128,
        "dimensionScale": 0.125,
    },
    "decodeTokenCount": 1,
    "decodeStartPosition": 141,
    "prefillTokenCount": 130,
    "prefillStartPosition": 11,
    "expected": {
        "decodeTokenIndex": 0,
        "decodeAbsoluteTokenPosition": 141,
        "decodeSamples": [
            {"layer": 0, "dimension": 0, "value": 752.0},
            {"layer": 0, "dimension": 128, "value": 768.0},
            {"layer": 1, "dimension": 257, "value": 66288.0},
            {"layer": 2, "dimension": 4095, "value": 131854.0},
        ],
        "prefillTokenIndex": 129,
        "prefillAbsoluteTokenPosition": 140,
        "historyTokenStart": 13,
        "historyLength": 128,
        "prefillSamples": [
            {"layer": 0, "dimension": 0, "value": 1057520.0},
            {"layer": 1, "dimension": 128, "value": 1123072.0},
            {"layer": 2, "dimension": 4095, "value": 1188622.0},
        ],
        "historySamples": [
            {"logical": 0, "layer": 0, "dimension": 0,
             "value": 17136.0},
            {"logical": 0, "layer": 1, "dimension": 128,
             "value": 82688.0},
            {"logical": 127, "layer": 2, "dimension": 4095,
             "value": 1188622.0},
        ],
    },
}


PROPOSAL_TOKEN_LAYOUT_FIXTURE = {
    "lastTargetPosition": 128,
    "pendingTokenId": 17,
    "noiseTokenId": 128799,
    "blockSize": 5,
    "expected": {
        "inputTokenIds": [17, 128799, 128799, 128799, 128799],
        "inputPositions": [129, 130, 131, 132, 133],
        "proposedOutputPositions": [130, 131, 132, 133, 134],
    },
}


DIRECT_CONTEXT_KV_FIXTURE = {
    "mainX": [[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, -3.0]],
    "projectionGenerator": {
        "shape": [512, 4],
        "kind": "repeatingIdentityRows",
    },
    "normWeight": {"shape": [512], "fill": 1.0},
    "absolutePositions": [0, 129],
    "normEps": 1.0e-6,
    "ropeTheta": 10000.0,
    "expected": {
        "normalizedSamples": [
            {"token": 0, "dimension": 0, "value": 0.365234375},
            {"token": 0, "dimension": 3, "value": 1.4609375},
            {"token": 1, "dimension": 0, "value": 1.0625},
            {"token": 1, "dimension": 3, "value": -1.5859375},
        ],
        "ropedSamples": [
            {"token": 0, "dimension": 448, "value": 0.365234375},
            {"token": 0, "dimension": 511, "value": 1.4609375},
            {"token": 1, "dimension": 448, "value": -1.1484375},
            {"token": 1, "dimension": 449, "value": 0.31640625},
            {"token": 1, "dimension": 510, "value": 0.29296875},
            {"token": 1, "dimension": 511, "value": -1.578125},
        ],
        "storedSamples": [
            {"token": 0, "dimension": 0, "value": 0.375},
            {"token": 0, "dimension": 1, "value": 0.75},
            {"token": 0, "dimension": 2, "value": 1.125},
            {"token": 0, "dimension": 3, "value": 1.5},
            {"token": 1, "dimension": 0, "value": 1.0},
            {"token": 1, "dimension": 1, "value": -0.5},
            {"token": 1, "dimension": 2, "value": 0.25},
            {"token": 1, "dimension": 3, "value": -1.625},
            {"token": 1, "dimension": 448, "value": -1.1484375},
            {"token": 1, "dimension": 511, "value": -1.578125},
        ],
        "nonropeScales": [
            [0.00390625, 0.00390625, 0.00390625, 0.00390625,
             0.00390625, 0.00390625, 0.00390625],
            [0.00390625, 0.00390625, 0.00390625, 0.00390625,
             0.00390625, 0.00390625, 0.00390625],
        ],
    },
}


def _dense_weight(
    rows: int,
    columns: int,
    *,
    row_multiplier: int,
    column_multiplier: int,
    modulus: int,
    offset: int,
    divisor: float,
    zero_replacement: float,
) -> list[list[float]]:
    """Declare compact deterministic inputs; never derive expected outputs."""

    matrix: list[list[float]] = []
    for row in range(rows):
        values: list[float] = []
        for column in range(columns):
            raw = (
                (row * row_multiplier + column * column_multiplier) % modulus
            ) - offset
            values.append((zero_replacement if raw == 0 else raw) / divisor)
        matrix.append(values)
    return matrix


HC_FUNCTION = _dense_weight(
    24, 8,
    row_multiplier=7,
    column_multiplier=5,
    modulus=19,
    offset=9,
    divisor=16.0,
    zero_replacement=0.5,
)
HC_HEAD_FUNCTION = _dense_weight(
    4, 8,
    row_multiplier=11,
    column_multiplier=3,
    modulus=23,
    offset=11,
    divisor=18.0,
    zero_replacement=0.75,
)


def _stage_setup_fixture() -> dict[str, Any]:
    denominator = math.sqrt(4153.0 + 1.0e-6)
    return {
        "targetHidden": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "mainProjection": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-2.0, 1.0, -1.0, 3.0, 2.0, -4.0],
        ],
        "mainNormWeight": [1.5, 0.75],
        "pendingEmbedding": [0.1, 0.2],
        "noiseEmbedding": [-0.3, 0.4],
        "blockSize": 5,
        "hcLanes": 4,
        "expected": {
            "concatenated": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "preNorm": [91.0, -5.0],
            "mainHidden": [136.5 / denominator, -3.75 / denominator],
            "draftRows": [
                [0.1, 0.2],
                [-0.3, 0.4],
                [-0.3, 0.4],
                [-0.3, 0.4],
                [-0.3, 0.4],
            ],
        },
    }


def _hc_fixture() -> dict[str, Any]:
    return {
        "hidden": [[1.25, -0.75], [-2.0, 0.5], [0.25, 1.75], [3.0, -1.25]],
        "function": HC_FUNCTION,
        "scale": [0.7, -1.1, 0.45],
        "base": [(index - 11.5) / 13.0 for index in range(24)],
        "iterations": 20,
        "normEps": 1.0e-6,
        "hcEps": 1.0e-6,
        "branchOutput": [0.6, -0.9],
        "headFunction": HC_HEAD_FUNCTION,
        "headScale": [0.65],
        "headBase": [-0.3, 0.1, 0.4, -0.2],
        "expected": {
            "pre": [
                0.2040499900619359,
                0.41126150490314217,
                0.3100615566979229,
                0.27319827233329325,
            ],
            "post": [
                0.6532702003342632,
                0.6949119756462091,
                0.6398549674229667,
                0.661712262339485,
            ],
            "combination": [
                [0.28042290941281844, 0.30546994564443514,
                 0.18879370687457867, 0.2253124380681679],
                [0.12982070068726406, 0.22995537117678608,
                 0.33408296668143966, 0.30613996145451017],
                [0.2691069076980839, 0.2577270529295607,
                 0.2830673713558346, 0.1900976680165209],
                [0.32064848220183373, 0.2068466302492181,
                 0.19405495508814724, 0.27844893246080105],
            ],
            "reduced": [0.329649683945496, 0.2537031437098677],
            "postOutput": [
                [0.8546866066008592, -0.5967769500929304],
                [1.1212529449271238, -0.4058383781342616],
                [0.8459023561057847, -0.3910903101373931],
                [1.2680052358119869, -0.7410690768040464],
            ],
            "head": [1.0130180313582116, 0.8355590801245076],
        },
    }


def _raw_cache_fixture() -> dict[str, Any]:
    return {
        "stageCount": 3,
        "window": 128,
        "width": 512,
        "startPosition": 0,
        "prefillTokenCount": 128,
        "rowGenerator": {
            "stageScale": 100000.0,
            "positionScale": 1000.0,
            "dimensionScale": 1.0,
        },
        "proposalPosition": 129,
        "draftGenerator": {
            "base": 500000.0,
            "stageScale": 100000.0,
            "slotScale": 1000.0,
            "dimensionScale": 1.0,
        },
        "expected": {
            "beforeWrapTokenStart": 0,
            "beforeWrapLength": 128,
            "beforeWrapSamples": [
                {"stage": 0, "logical": 0, "dimension": 0, "value": 0.0},
                {"stage": 1, "logical": 1, "dimension": 7,
                 "value": 101007.0},
                {"stage": 2, "logical": 127, "dimension": 511,
                 "value": 327511.0},
            ],
            "after128TokenStart": 1,
            "after128Samples": [
                {"stage": 0, "logical": 0, "dimension": 0,
                 "value": 1000.0},
                {"stage": 2, "logical": 127, "dimension": 511,
                 "value": 328511.0},
                {"stage": 0, "physical": 0, "dimension": 0,
                 "value": 128000.0},
                {"stage": 2, "physical": 127, "dimension": 511,
                 "value": 327511.0},
            ],
            "proposalSamples": [
                {"stage": 0, "view": 0, "dimension": 0,
                 "value": 1000.0},
                {"stage": 1, "view": 126, "dimension": 7,
                 "value": 227007.0},
                {"stage": 2, "view": 127, "dimension": 511,
                 "value": 328511.0},
                {"stage": 0, "view": 128, "dimension": 0,
                 "value": 500000.0},
                {"stage": 2, "view": 132, "dimension": 511,
                 "value": 704511.0},
            ],
            "verifierTokenCount": 4,
            "acceptedRows": 3,
            "afterCommitTokenStart": 4,
            "afterCommitSamples": [
                {"stage": 0, "logical": 0, "dimension": 0,
                 "value": 4000.0},
                {"stage": 2, "logical": 127, "dimension": 511,
                 "value": 331511.0},
                {"stage": 2, "physical": 3, "dimension": 511,
                 "value": 331511.0},
            ],
        },
    }


def _stage_chain_fixture() -> dict[str, Any]:
    return {
        "draftHidden": [[1.0, 2.0], [-1.0, 0.5]],
        "mainHidden": [0.25, -0.75],
        "stageWeights": [
            [[1.0, 0.5], [-0.25, 1.5]],
            [[0.2, -1.0], [1.25, 0.3]],
            [[-0.5, 0.75], [0.4, 1.1]],
        ],
        "mainWeights": [
            [[0.6, -0.2], [0.1, 0.8]],
            [[-0.3, 0.5], [0.7, -0.4]],
            [[0.9, 0.2], [-0.6, 0.3]],
        ],
        "stageBiases": [[0.1, -0.2], [0.3, 0.05], [-0.15, 0.25]],
        "expectedStageOutputs": [
            [[2.4, 1.9749999999999999],
             [-0.35, 0.22499999999999992]],
            [[-1.6449999999999998, 4.1175],
             [-0.4449999999999999, 0.15499999999999997]],
            [[3.835625, 3.7462500000000003],
             [0.26374999999999993, -0.1325]],
        ],
    }


def _draft_head_fixture() -> dict[str, Any]:
    return {
        "finalHC": [
            [[0.5, -1.0], [-1.2, 0.7], [1.8, 0.4], [-0.3, 2.1]],
            [[0.9, -0.8], [-1.1, 1.0], [1.6, -0.1], [0.3, 2.0]],
            [[1.3, -0.6], [-1.0, 1.3], [1.4, -0.6], [0.9, 1.9]],
            [[1.7, -0.4], [-0.9, 1.6], [1.2, -1.1], [1.5, 1.8]],
            [[2.1, -0.2], [-0.8, 1.9], [1.0, -1.6], [2.1, 1.7]],
        ],
        "firstPreviousToken": 0,
        "outputProjection": [
            [0.8, -0.3], [-0.5, 1.1], [0.4, 0.9], [-1.2, -0.2]
        ],
        "normWeight": [1.3, 0.7],
        "hcFunction": HC_HEAD_FUNCTION,
        "hcScale": [0.65],
        "hcBase": [-0.3, 0.1, 0.4, -0.2],
        "markovEmbedding": [
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -1.0]
        ],
        "markovProjection": [
            [0.0, 3.0], [3.0, 0.0], [0.4, 0.2], [-1.0, -1.0]
        ],
        "confidenceProjection": [0.45, -0.35, 0.8, -0.6],
        "confidenceThreshold": 0.5,
        "expected": {
            "hidden": [
                [1.2936834687982695, 0.9911636495090587],
                [1.6714235080645645, 0.8556589291733565],
                [1.9771675600283452, 0.7402223647146006],
                [2.237601153425474, 0.6371488719862743],
                [2.489925748628722, 0.5359515985803998],
            ],
            "baseLogits": [
                [0.986890202487156, -0.06742332764063563,
                 1.12561189725402, -1.8716766536419596],
                [1.1738644825355482, -0.32202574740519424,
                 1.0605996734743488, -2.0540194199312016],
                [1.2732856785955262, -0.47907966256549367,
                 1.0010922430068543, -2.1355400535227265],
                [1.3332206726172922, -0.5858769578517731,
                 0.9512736356628775, -2.1760511466214147],
                [1.3753559263777277, -0.6695113697575032,
                 0.9064071559998286, -2.1984376755924266],
            ],
            "tokens": [1, 0, 1, 0, 1],
            "correctedLogits": [
                [0.986890202487156, 2.9325766723593643,
                 1.52561189725402, -2.8716766536419596],
                [4.173864482535548, -0.32202574740519424,
                 1.2605996734743488, -3.0540194199312016],
                [1.2732856785955262, 2.5209203374345064,
                 1.4010922430068544, -3.1355400535227265],
                [4.333220672617292, -0.5858769578517731,
                 1.1512736356628774, -3.1760511466214147],
                [1.3753559263777277, 2.3304886302424968,
                 1.3064071559998287, -3.1984376755924266],
            ],
            "confidenceLogits": [
                1.0352502836310506,
                -0.1473400465816207,
                1.430647574362645,
                0.18391841384626717,
                1.732883527379785,
            ],
            "confidenceKeep": 1,
        },
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
        "support": {
            "general.architecture": "deepseek4-dspark",
            "general.name": "DeepSeek V4 Flash DSpark support",
            "general.alignment": 32,
            "dspark.block_size": 5,
            "dspark.markov_rank": 256,
            "dspark.noise_token_id": 128799,
            "dspark.target_layer_ids": [40, 41, 42],
            "dspark.stage_count": 3,
            "dspark.n_layers": 3,
            "dspark.source.revision":
                "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "dspark.source.config_sha256":
                "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
            "dspark.source.index_sha256":
                "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
            "dspark.source.shard46_sha256":
                "5db924ca907e0d93acd975bd5079c3662717f9ac709f23d079bd8f816d29d9dd",
            "dspark.source.shard47_sha256":
                "62816173f9f6e136b20b48e3b6f16613ac9ea02b5603f636928b253244a548bd",
            "dspark.source.shard48_sha256":
                "cc43742bd24ae6bcdea343a91442f6f66aed2cfebcc6b235470204851ce2f8a9",
        },
    }


def build_fixture() -> dict[str, Any]:
    return {
        "schemaVersion": 3,
        "profile": "deepseek-v4-flash-0731-dspark",
        "generatedBy": "tools/dspark_oracle/generate_fixtures.py",
        "numericType": "float64",
        "cases": {
            "postLayerHCMean": HC_CAPTURE_FIXTURE,
            "targetCaptureRows": TARGET_CAPTURE_FIXTURE,
            "proposalTokenLayout": PROPOSAL_TOKEN_LAYOUT_FIXTURE,
            "stageSetup": _stage_setup_fixture(),
            "directContextKV": DIRECT_CONTEXT_KV_FIXTURE,
            "hyperConnection": _hc_fixture(),
            "stageChain": _stage_chain_fixture(),
            "rawCache": _raw_cache_fixture(),
            "draftHead": _draft_head_fixture(),
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
            "markovSampled": {
                "baseLogits": BASE_LOGITS,
                "firstPreviousToken": 2,
                "embedding": EMBEDDING,
                "projection": MARKOV_PROJECTION,
                "temperature": 0.7,
                "samplingUniforms": [0.2, 0.65, 0.9],
                "expected": {
                    "tokens": [0, 2, 3],
                    "correctedLogits": [
                        [1.1, 0.39999999999999997, -0.4, -0.4],
                        [0.4, -0.30000000000000004, 0.1,
                         -0.30000000000000004],
                        [-0.4, 0.09999999999999998, 1.0,
                         -0.10000000000000003],
                    ],
                    "probabilities": [
                        [0.6240180400017246, 0.2295634078367331,
                         0.07320927608077114, 0.07320927608077114],
                        [0.41890116579641756, 0.15410512677925176,
                         0.2728885806450787, 0.15410512677925176],
                        [0.08356420606463894, 0.17069886583353602,
                         0.6174606064742179, 0.12827632162760708],
                    ],
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
