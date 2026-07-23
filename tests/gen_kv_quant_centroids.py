#!/usr/bin/env python3
"""Generate/check the standard-normal Lloyd-Max TQ4 centroid table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ds4_kv_quant.c"
METAL_SOURCE = ROOT / "metal" / "qwen35.metal"
BEGIN = "/* BEGIN GENERATED DS4_KV_TQ4_CENTROIDS */"
END = "/* END GENERATED DS4_KV_TQ4_CENTROIDS */"
METAL_BEGIN = "/* BEGIN GENERATED DS4_KV_TQ4_METAL_CENTROIDS */"
METAL_END = "/* END GENERATED DS4_KV_TQ4_METAL_CENTROIDS */"


def gaussian_pdf(value: float) -> float:
    return math.exp(-value * value / 2.0) / math.sqrt(2.0 * math.pi)


def trapezoid(first: float, last: float, weighted: bool) -> float:
    count = 200
    step = (last - first) / count

    def sample(value: float) -> float:
        density = gaussian_pdf(value)
        return value * density if weighted else density

    total = 0.5 * (sample(first) + sample(last))
    total += sum(sample(first + i * step) for i in range(1, count))
    return total * step


def solve() -> list[float]:
    level_count = 16
    low = -3.5
    high = 3.5
    centroids = [
        low + (high - low) * (i + 0.5) / level_count
        for i in range(level_count)
    ]
    for _ in range(200):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0
            for i in range(level_count - 1)
        ]
        edges = [low * 3.0, *boundaries, high * 3.0]
        updated = []
        for i in range(level_count):
            denominator = trapezoid(edges[i], edges[i + 1], False)
            numerator = trapezoid(edges[i], edges[i + 1], True)
            updated.append(
                numerator / denominator
                if denominator > 1.0e-15
                else centroids[i]
            )
        if max(abs(a - b) for a, b in zip(updated, centroids)) < 1.0e-10:
            centroids = updated
            break
        centroids = updated
    return centroids


def generated_block() -> str:
    values = solve()
    rows = []
    for start in range(0, len(values), 4):
        row = ", ".join(f"{value: .9f}f" for value in values[start : start + 4])
        rows.append(f"    {row},")
    return "\n".join(
        [
            BEGIN,
            "static const float ds4_kv_tq4_centroid_std_normal[16] = {",
            *rows,
            "};",
            END,
        ]
    )


def generated_metal_block() -> str:
    values = solve()
    rows = []
    for start in range(0, len(values), 4):
        row = ", ".join(f"{value: .9f}f" for value in values[start : start + 4])
        rows.append(f"    {row},")
    return "\n".join(
        [
            METAL_BEGIN,
            "constant float qwen35_tq4_centroid_std_normal[16] = {",
            *rows,
            "};",
            METAL_END,
        ]
    )


def replace_generated(
    source: str,
    begin: str,
    end: str,
    expected: str,
) -> tuple[str, bool]:
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + expected + source[finish:], source[start:finish] == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if ds4_kv_quant.c differs from the generated table",
    )
    args = parser.parse_args()

    source = SOURCE.read_text(encoding="utf-8")
    metal_source = METAL_SOURCE.read_text(encoding="utf-8")
    updated, source_matches = replace_generated(
        source, BEGIN, END, generated_block()
    )
    updated_metal, metal_matches = replace_generated(
        metal_source, METAL_BEGIN, METAL_END, generated_metal_block()
    )
    if args.check:
        if not source_matches or not metal_matches:
            raise SystemExit(
                "KV centroid table is stale; "
                "run tests/gen_kv_quant_centroids.py"
            )
        return 0

    SOURCE.write_text(updated, encoding="utf-8")
    METAL_SOURCE.write_text(updated_metal, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
