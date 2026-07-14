#!/usr/bin/env python3
"""Offline synthetic tests for the Qwen model-backed logits comparator."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_logits


@dataclass(frozen=True)
class SyntheticFixture:
    root: Path
    model: Path
    ds4_logits: Path
    llama_logits: Path
    llama_prompt: Path
    llama_log: Path
    ds4_tokens: Path
    llama_source: Path


def generate_synthetic_fixture(root: Path, *, binary: bool = True) -> SyntheticFixture:
    """Generate a complete fake comparison case without a model or network."""
    model = root / "qwen-synthetic.gguf"
    model.write_bytes(b"synthetic model identity only\n")

    ds4_values = [-20.0] * compare_logits.QWEN_VOCAB
    llama_values = [-20.0] * compare_logits.QWEN_VOCAB
    for token_id in range(80):
        ds4_values[token_id] = 100.0 - token_id
        llama_values[token_id] = 100.0 - token_id + (0.01 if token_id % 2 else -0.01)

    # Padding is deliberately larger than every real token.  The expected
    # top-1 remains token zero and the padding differences do not affect RMSE.
    for token_id in range(compare_logits.QWEN_PAD_START, compare_logits.QWEN_PAD_END):
        ds4_values[token_id] = 1000.0 + token_id
        llama_values[token_id] = -1000.0 - token_id

    ds4_logits = root / "ds4-logits.json"
    ds4_logits.write_text(
        json.dumps(
            {
                "source": "ds4",
                "model": str(model.resolve()),
                "backend": "cpu",
                "quant_bits": 4,
                "prompt_tokens": 3,
                "ctx": 16,
                "vocab": compare_logits.QWEN_VOCAB,
                "argmax_token": {"id": 0, "text": "x", "bytes": [120]},
                "argmax_logit": ds4_values[0],
                "logits": ds4_values,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    suffix = ".bin" if binary else ".txt"
    llama_logits = root / f"llamacpp-{model.stem}{suffix}"
    if binary:
        with llama_logits.open("wb") as destination:
            for value in llama_values:
                destination.write(struct.pack("<f", value))
    else:
        llama_logits.write_text(
            "".join(f"{index}: {value}\n" for index, value in enumerate(llama_values)),
            encoding="utf-8",
        )

    llama_prompt = root / f"llamacpp-{model.stem}-prompt.txt"
    llama_prompt.write_text(
        "prompt: synthetic\n"
        "n_tokens: 3\n"
        "token ids: 248044, 42, 17\n",
        encoding="utf-8",
    )
    llama_log = root / "llama-debug.log"
    llama_log.write_text(
        "llama_model_loader: loaded meta data with 30 key-value pairs and "
        f"733 tensors from {model.resolve()} (version GGUF V3 (latest))\n",
        encoding="utf-8",
    )
    ds4_tokens = root / "ds4-tokens.txt"
    ds4_tokens.write_text(
        "[248044, 42, 17]\n248044  <|endoftext|>\n",
        encoding="utf-8",
    )
    llama_source = root / "llama.cpp"
    debug_dir = llama_source / "examples" / "debug"
    debug_dir.mkdir(parents=True)
    debug_dir.joinpath("debug.cpp").write_text(
        f"auto a = {compare_logits.LLAMA_PARSE_SPECIAL_CALL};\n"
        f"auto b = {compare_logits.LLAMA_PARSE_SPECIAL_CALL};\n",
        encoding="utf-8",
    )
    return SyntheticFixture(
        root=root,
        model=model,
        ds4_logits=ds4_logits,
        llama_logits=llama_logits,
        llama_prompt=llama_prompt,
        llama_log=llama_log,
        ds4_tokens=ds4_tokens,
        llama_source=llama_source,
    )


def pinned_git_revision() -> Any:
    baseline = (
        f"auto a = {compare_logits.LLAMA_UNPATCHED_CALL};\n"
        f"auto b = {compare_logits.LLAMA_UNPATCHED_CALL};\n"
    )

    def fake_run(command: list[str], **_: object) -> object:
        output = (
            compare_logits.LLAMA_CPP_REVISION + "\n"
            if command[-2:] == ["rev-parse", "HEAD"]
            else baseline
        )
        return compare_logits.subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=output,
            stderr="",
        )

    return mock.patch.object(compare_logits.subprocess, "run", side_effect=fake_run)


class LlamaFormatTests(unittest.TestCase):
    def test_binary_float32_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.bin"
            path.write_bytes(struct.pack("<4f", 1.25, -2.5, math.inf, math.nan))
            values = compare_logits.load_llama_binary(path)
            self.assertEqual(values[:2], (1.25, -2.5))
            self.assertTrue(math.isinf(values[2]))
            self.assertTrue(math.isnan(values[3]))

    def test_documented_text_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.txt"
            path.write_text("0: 1.25\n1: -2.5\n2: inf\n", encoding="utf-8")
            values = compare_logits.load_llama_text(path)
            self.assertEqual(values[:2], (1.25, -2.5))
            self.assertTrue(math.isinf(values[2]))

    def test_text_loader_rejects_non_contiguous_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.txt"
            path.write_text("0: 1\n2: 3\n", encoding="utf-8")
            with self.assertRaisesRegex(compare_logits.ComparisonError, "contiguous"):
                compare_logits.load_llama_text(path)


class ComparisonTests(unittest.TestCase):
    def test_complete_binary_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = generate_synthetic_fixture(Path(temporary))
            with pinned_git_revision():
                report = compare_logits.compare_files(
                    ds4_logits_path=fixture.ds4_logits,
                    llama_logits_path=fixture.llama_logits,
                    llama_prompt_path=fixture.llama_prompt,
                    llama_log_path=fixture.llama_log,
                    ds4_tokens_path=fixture.ds4_tokens,
                    model_path=fixture.model,
                    llama_source_path=fixture.llama_source,
                )

        self.assertEqual(report["schema"], "ds4-qwen-logits-comparison-v1")
        provenance = report["provenance"]
        self.assertEqual(provenance["prompt_tokens"], 3)
        self.assertEqual(provenance["logits_position"], 2)
        self.assertTrue(provenance["token_ids_match"])
        self.assertEqual(
            provenance["llama_cpp_revision"], compare_logits.LLAMA_CPP_REVISION
        )
        self.assertTrue(provenance["llama_debug_parse_special_patch"])
        self.assertEqual(len(provenance["llama_debug_source_sha256"]), 64)

        result = report["comparison"]
        coverage = result["finite_coverage"]
        self.assertEqual(coverage["eligible"], compare_logits.QWEN_EFFECTIVE_VOCAB)
        self.assertEqual(coverage["paired_finite"], compare_logits.QWEN_EFFECTIVE_VOCAB)
        self.assertEqual(
            coverage["excluded_qwen_padding_ids"],
            compare_logits.QWEN_VOCAB - compare_logits.QWEN_EFFECTIVE_VOCAB,
        )
        self.assertEqual(result["top_1"]["ds4_id"], 0)
        self.assertEqual(result["top_1"]["llama_id"], 0)
        self.assertTrue(result["top_1"]["match"])
        for count in compare_logits.TOP_K:
            self.assertEqual(result["top_k"][str(count)]["overlap"], count)
        self.assertLess(result["max_abs"], 0.011)
        self.assertGreater(result["cosine"], 0.999999)

    def test_padding_never_wins_selection(self) -> None:
        left = [-50.0] * compare_logits.QWEN_VOCAB
        right = [-50.0] * compare_logits.QWEN_VOCAB
        for token_id in range(64):
            left[token_id] = 64.0 - token_id
            right[token_id] = 64.0 - token_id
        left[compare_logits.QWEN_PAD_START] = 1.0e9
        right[compare_logits.QWEN_PAD_START + 1] = 1.0e9

        report = compare_logits.compare_vectors(left, right, ds4_reported_argmax=0)
        self.assertEqual(report["top_1"]["ds4_id"], 0)
        self.assertEqual(report["top_1"]["llama_id"], 0)
        selected = {
            token_id
            for count in compare_logits.TOP_K
            for token_id in report["top_k"][str(count)]["ds4_ids"]
            + report["top_k"][str(count)]["llama_ids"]
        }
        self.assertFalse(any(compare_logits.is_qwen_padding(token_id) for token_id in selected))

    def test_reported_padded_argmax_is_rejected(self) -> None:
        values = [float(-index) for index in range(80)]
        with self.assertRaisesRegex(compare_logits.ComparisonError, "padded Qwen token"):
            compare_logits.compare_vectors(
                values + [0.0] * (compare_logits.QWEN_PAD_START - len(values) + 1),
                values + [0.0] * (compare_logits.QWEN_PAD_START - len(values) + 1),
                ds4_reported_argmax=compare_logits.QWEN_PAD_START,
            )

    def test_finite_coverage_counts_each_source_and_pairs(self) -> None:
        left = [100.0 - index for index in range(80)]
        right = left.copy()
        left[70] = math.nan
        right[71] = math.inf
        report = compare_logits.compare_vectors(left, right, ds4_reported_argmax=0)
        coverage = report["finite_coverage"]
        self.assertEqual(coverage["eligible"], 80)
        self.assertEqual(coverage["ds4_finite"], 79)
        self.assertEqual(coverage["llama_finite"], 79)
        self.assertEqual(coverage["paired_finite"], 78)
        self.assertEqual(coverage["paired_fraction"], 78 / 80)

    def test_prompt_token_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = generate_synthetic_fixture(Path(temporary))
            fixture.ds4_tokens.write_text("[248044, 42, 18]\n", encoding="utf-8")
            with pinned_git_revision(), self.assertRaisesRegex(
                compare_logits.ComparisonError, "token mismatch"
            ):
                compare_logits.compare_files(
                    ds4_logits_path=fixture.ds4_logits,
                    llama_logits_path=fixture.llama_logits,
                    llama_prompt_path=fixture.llama_prompt,
                    llama_log_path=fixture.llama_log,
                    ds4_tokens_path=fixture.ds4_tokens,
                    model_path=fixture.model,
                    llama_source_path=fixture.llama_source,
                )

    def test_model_output_name_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = generate_synthetic_fixture(Path(temporary))
            renamed = fixture.llama_logits.with_name("llamacpp-wrong.bin")
            fixture.llama_logits.rename(renamed)
            with pinned_git_revision(), self.assertRaisesRegex(
                compare_logits.ComparisonError, "filename"
            ):
                compare_logits.compare_files(
                    ds4_logits_path=fixture.ds4_logits,
                    llama_logits_path=renamed,
                    llama_prompt_path=fixture.llama_prompt,
                    llama_log_path=fixture.llama_log,
                    ds4_tokens_path=fixture.ds4_tokens,
                    model_path=fixture.model,
                    llama_source_path=fixture.llama_source,
                )

    def test_unpatched_llama_debug_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            debug_dir = source / "examples" / "debug"
            debug_dir.mkdir(parents=True)
            debug_dir.joinpath("debug.cpp").write_text(
                "auto tokens = common_tokenize(ctx, params.prompt, add_bos);\n",
                encoding="utf-8",
            )
            with pinned_git_revision(), self.assertRaisesRegex(
                compare_logits.ComparisonError, "exactly the two documented"
            ):
                compare_logits.validate_llama_source(source)


if __name__ == "__main__":
    unittest.main()
