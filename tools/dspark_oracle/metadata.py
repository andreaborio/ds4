"""Fail-closed semantic metadata validation for the final 0731 checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


SCHEMA_PATH = Path(__file__).with_name("schema.json")


class MetadataError(ValueError):
    """The supplied metadata does not identify the pinned 0731 DSpark model."""


@dataclass(frozen=True)
class DSparkMetadata:
    block_size: int
    markov_rank: int
    noise_token_id: int
    stage_count: int
    target_layer_ids: tuple[int, ...]
    target_layer_count: int
    vocab_size: int
    expert_count: int
    experts_per_token: int


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    try:
        document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot load DSpark oracle schema: {exc}") from exc
    if not isinstance(document, dict) or document.get("schemaVersion") != 1:
        raise MetadataError("DSpark oracle schemaVersion must be 1")
    if document.get("profile") != "deepseek-v4-flash-0731-dspark":
        raise MetadataError("DSpark oracle schema has the wrong profile")
    return document


def _same_typed_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_same_typed_value(item, want) for item, want in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def validate_0731_metadata(
    metadata: Mapping[str, Any], *, flavor: str
) -> DSparkMetadata:
    """Validate either official HF config or canonical embedded-GGUF metadata.

    Extra keys are allowed because both containers carry unrelated target-model
    metadata.  Required DSpark identity keys have exactly one canonical spelling
    per flavor; this oracle intentionally does not define compatibility aliases.
    """

    if not isinstance(metadata, Mapping):
        raise MetadataError("metadata must be a mapping")
    schema = load_schema()
    expected_by_flavor = schema.get("metadata", {})
    field_maps = schema.get("fieldMap", {})
    if flavor not in expected_by_flavor or flavor not in field_maps:
        raise MetadataError(f"unsupported metadata flavor: {flavor!r}")

    errors: list[str] = []
    expected = expected_by_flavor[flavor]
    if flavor == "hf":
        unknown_dspark = sorted(
            key
            for key in metadata
            if key.startswith("dspark_") and key not in expected
        )
    else:
        unknown_dspark = sorted(
            key
            for key in metadata
            if (
                key.startswith("dspark.")
                or key.startswith("deepseek4.dspark")
            )
            and key not in expected
        )
    if unknown_dspark:
        errors.append("unknown DSpark keys: " + ", ".join(unknown_dspark))
    for key, wanted in expected.items():
        if key not in metadata:
            errors.append(f"missing {key}")
        elif not _same_typed_value(metadata[key], wanted):
            errors.append(f"{key} must be {wanted!r}, got {metadata[key]!r}")
    if errors:
        raise MetadataError("0731 metadata mismatch: " + "; ".join(errors))

    semantic = schema["semantic"]
    mapping = field_maps[flavor]

    def value(name: str) -> Any:
        key = mapping.get(name)
        return semantic[name] if key is None else metadata[key]

    result = DSparkMetadata(
        block_size=int(value("blockSize")),
        markov_rank=int(value("markovRank")),
        noise_token_id=int(value("noiseTokenId")),
        stage_count=int(value("stageCount")),
        target_layer_ids=tuple(int(item) for item in value("targetLayerIds")),
        target_layer_count=int(value("targetLayerCount")),
        vocab_size=int(value("vocabSize")),
        expert_count=int(value("expertCount")),
        experts_per_token=int(value("expertsPerToken")),
    )
    if len(result.target_layer_ids) != result.stage_count:
        raise MetadataError("target_layer_ids count must equal stage_count")
    if flavor == "gguf" and metadata["dspark.n_layers"] != result.stage_count:
        raise MetadataError("dspark.n_layers must equal dspark.stage_count")
    if tuple(sorted(set(result.target_layer_ids))) != result.target_layer_ids:
        raise MetadataError("target_layer_ids must be unique and strictly increasing")
    if any(layer < 0 or layer >= result.target_layer_count for layer in result.target_layer_ids):
        raise MetadataError("target_layer_ids contains a layer outside the target")
    if result.noise_token_id < 0 or result.noise_token_id >= result.vocab_size:
        raise MetadataError("noise_token_id is outside the target vocabulary")
    return result
