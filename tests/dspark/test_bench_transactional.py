#!/usr/bin/env python3
"""Fail-closed structural gate for ds4-bench generation transactions."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "ds4_bench.c"


def _blank_noncode(source: str) -> str:
    """Blank comments and literals while preserving offsets and newlines."""

    pattern = re.compile(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.DOTALL,
    )

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return pattern.sub(blank, source)


def _function(source: str, name: str) -> str:
    clean = _blank_noncode(source)
    match = re.search(rf"\b{name}\s*\(", clean)
    if not match:
        raise AssertionError(f"missing function {name}")
    open_brace = clean.find("{", match.end())
    if open_brace < 0:
        raise AssertionError(f"missing body for {name}")
    depth = 0
    for index in range(open_brace, len(clean)):
        if clean[index] == "{":
            depth += 1
        elif clean[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"unterminated body for {name}")


def _compact(source: str) -> str:
    return re.sub(r"\s+", "", _blank_noncode(source))


def validate_transactional_bench(source: str) -> None:
    worker = _compact(_function(source, "run_transactional_greedy_decode"))
    aborter = _compact(_function(source, "abort_transactional_generation_block"))
    main = _compact(_function(source, "main"))
    whole = _compact(source)

    required_worker = (
        "ds4_generation_rngrng={.state=0,.position=0};",
        "while(token_count<gen_tokens)",
        ".temperature=0.0f",
        ".top_k=1",
        ".top_p=1.0f",
        ".min_p=0.0f",
        ".max_output_tokens=gen_tokens-token_count",
        "block.count>DS4_GENERATION_BLOCK_MAX_TOKENS",
        "block.count>(uint32_t)request.max_output_tokens",
        "abort_transactional_generation_block(session,block.cookie,0,&request.rng);",
        "uint32_tobserved_count=0;",
        "observed_count++;",
        ".adopted_count=block.count",
        ".observed_count=block.count",
        ".mode=DS4_GENERATION_COMMIT_RETAIN",
        "rng.state!=0||rng.position!=(uint64_t)committed_count",
        ".max_output_tokens=0",
        "flush_block.cookie!=0||flush_block.count!=0",
        "ds4_session_pos(session)!=initial_pos+token_count",
        "flush_per_token_ms=flush_ms/(double)token_count",
        "token_ms[i]+=flush_per_token_ms",
    )
    for fragment in required_worker:
        if fragment not in worker:
            raise AssertionError(f"missing transactional benchmark contract: {fragment}")
    for fragment in (
        ".temperature=0.0f",
        ".top_k=1",
        ".top_p=1.0f",
        ".min_p=0.0f",
    ):
        if worker.count(fragment) != 2:
            raise AssertionError(
                f"both generation and flush requests must be exact greedy: {fragment}"
            )

    if worker.count("ds4_session_generation_block_begin(") != 2:
        raise AssertionError("decode must have one block begin plus one final flush begin")
    if worker.count("ds4_session_generation_block_commit(") != 1:
        raise AssertionError("the decode worker must have one singular RETAIN site")
    if worker.count("abort_transactional_generation_block(") != 3:
        raise AssertionError(
            "invalid block/token and failed RETAIN must share one abort-ledger owner"
        )
    if worker.count(
        "abort_transactional_generation_block(session,block.cookie,"
        "observed_count,&request.rng);"
    ) != 2:
        raise AssertionError(
            "invalid-token and failed-RETAIN paths must publish their observations"
        )
    required_aborter = (
        ".adopted_count=0",
        ".observed_count=observed_count",
        ".mode=DS4_GENERATION_COMMIT_INVALIDATE",
        "ds4_generation_rngabort_rng=*rng_start",
        "ds4_session_generation_block_commit(",
        "ds4_session_invalidate(session);",
    )
    for fragment in required_aborter:
        if fragment not in aborter:
            raise AssertionError(f"missing fail-closed abort ledger contract: {fragment}")
    if worker.count("ds4_session_invalidate(session);") < 4:
        raise AssertionError("begin/ledger/flush failures without an open cookie must invalidate")

    forbidden_worker = (
        "ds4_session_argmax(",
        "ds4_session_argmax_excluding(",
        "ds4_session_sample(",
        "ds4_session_eval(",
        "ds4_session_eval_speculative_argmax(",
        "ds4_token_eos(",
    )
    for fragment in forbidden_worker:
        if fragment in worker:
            raise AssertionError(f"legacy or EOS-terminal decode path is reachable: {fragment}")
    if "ds4_session_argmax_excluding(" in whole:
        raise AssertionError("ds4-bench must use ordinary greedy argmax, including EOS")

    required_timing = (
        "constdoubleblock_t0=bench_now_sec();",
        "constdoubleblock_ready=bench_now_sec();",
        "*first_token_ready_sec_out=block_ready-generation_start_sec;",
        "(bench_now_sec()-block_t0)*1000.0/(double)block.count",
        "constdoubleflush_t0=bench_now_sec();",
        "constdoubleflush_ms=(bench_now_sec()-flush_t0)*1000.0;",
    )
    for fragment in required_timing:
        if fragment not in worker:
            raise AssertionError(f"missing block-aware timing contract: {fragment}")

    required_main = (
        "constdoublegen_t0=bench_now_sec();",
        "run_transactional_greedy_decode(",
        "constdoublegen_t1=bench_now_sec();",
        "decode_token_count!=cfg.gen_tokens",
    )
    positions = []
    for fragment in required_main:
        index = main.find(fragment)
        if index < 0:
            raise AssertionError(f"missing benchmark integration contract: {fragment}")
        positions.append(index)
    if positions[:3] != sorted(positions[:3]):
        raise AssertionError("the final flush must remain inside the generation timer")


class TransactionalBenchStructuralTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = BENCH_PATH.read_text(encoding="utf-8")

    def test_current_source_satisfies_contract(self) -> None:
        validate_transactional_bench(self.source)

    def test_adversarial_mutations_fail_closed(self) -> None:
        mutations = {
            "sampled instead of greedy": (".temperature = 0.0f", ".temperature = 1.0f"),
            "greedy RNG draw accepted": ("rng.state != 0", "rng.state != UINT64_MAX"),
            "partial adoption": (".adopted_count = block.count", ".adopted_count = 0"),
            "unobserved suffix": (".observed_count = block.count", ".observed_count = 0"),
            "non-retained block": (
                ".mode = DS4_GENERATION_COMMIT_RETAIN",
                ".mode = DS4_GENERATION_COMMIT_INVALIDATE",
            ),
            "flush opens output": (".max_output_tokens = 0,", ".max_output_tokens = 1,"),
            "flush frontier unchecked": (
                "ds4_session_pos(session) != initial_pos + token_count",
                "ds4_session_pos(session) < initial_pos",
            ),
            "legacy EOS exclusion": (
                "run_transactional_greedy_decode(",
                "ds4_session_argmax_excluding(session, 0); run_transactional_greedy_decode(",
            ),
            "block cost not amortized": (
                "/ (double)block.count",
                "/ 1.0",
            ),
            "final flush outside timer": (
                "const double gen_t1 = bench_now_sec();",
                "const double gen_t1 = gen_t0;",
            ),
            "observed ledger not advanced": (
                "observed_count++;",
                "observed_count += 0;",
            ),
            "invalid block bypasses abort ledger": (
                "abort_transactional_generation_block(\n                session, block.cookie, 0, &request.rng);",
                "ds4_session_invalidate(session);",
            ),
            "invalid token abort does not publish observations": (
                ".observed_count = observed_count,",
                ".observed_count = 0,",
            ),
            "failed retain does not invalidate transaction": (
                "abort_transactional_generation_block(\n                session, block.cookie, observed_count, &request.rng);",
                "ds4_session_invalidate(session);",
            ),
        }
        for label, (old, new) in mutations.items():
            with self.subTest(label=label):
                self.assertIn(old, self.source)
                mutated = self.source.replace(old, new, 1)
                with self.assertRaises(AssertionError):
                    validate_transactional_bench(mutated)


if __name__ == "__main__":
    unittest.main()
