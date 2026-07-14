#!/usr/bin/env python3
"""Generate ds4's exact Qwen3.6 Unicode tables.

Normal generation and --check are offline: they consume the small, committed
semantic cache in tests/qwen/qwen_unicode_ucd_cache.txt.  --refresh-cache is a
maintainer operation that requires the two pinned official UCD.zip files and
verifies their SHA-256 digests before replacing the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "tests/qwen/qwen_unicode_ucd_cache.txt"
DEFAULT_OUTPUT = ROOT / "ds4_qwen_unicode_data.inc"

UCD9_SHA256 = "df9e028425816fd5117eaea7173704056f88f7cd030681e457c6f3827f9390ec"
UCD16_SHA256 = "c86dd81f2b14a43b0cc064aa5f89aa7241386801e35c59c7984e579832634eb2"
EXPECTED_CACHE_SHA256 = "400d9a7d10217d81727248529d2297e47da60ef2451b654b68c0fead4528a88e"
MAX_CODEPOINT = 0x110000

CLASS_OTHER = 0
CLASS_LETTER = 1
CLASS_MARK = 2
CLASS_NUMBER = 3


@dataclass(frozen=True)
class SourceData:
    category_ranges: tuple[tuple[int, int, int], ...]
    whitespace_ranges: tuple[tuple[int, int], ...]
    ccc_ranges: tuple[tuple[int, int, int], ...]
    decompositions: tuple[tuple[int, tuple[int, ...]], ...]
    composition_exclusions: frozenset[int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_text(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} in archive, found {matches}")
    return zf.read(matches[0]).decode("utf-8")


def parse_unicode_data(text: str) -> tuple[list[str], list[int], dict[int, tuple[int, ...]]]:
    categories = ["Cn"] * MAX_CODEPOINT
    ccc = [0] * MAX_CODEPOINT
    decompositions: dict[int, tuple[int, ...]] = {}
    first: tuple[int, str, int, str] | None = None

    for raw in text.splitlines():
        fields = raw.split(";")
        if len(fields) < 6:
            raise ValueError(f"bad UnicodeData record: {raw!r}")
        cp = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        combining = int(fields[3])
        decomposition = fields[5]

        if name.endswith(", First>"):
            if first is not None:
                raise ValueError("nested UnicodeData First record")
            first = (cp, category, combining, decomposition)
            continue
        if name.endswith(", Last>"):
            if first is None:
                raise ValueError("UnicodeData Last without First")
            lo, first_category, first_ccc, first_decomposition = first
            if (category, combining, decomposition) != (
                    first_category, first_ccc, first_decomposition):
                raise ValueError("UnicodeData First/Last metadata mismatch")
            for value in range(lo, cp + 1):
                categories[value] = category
                ccc[value] = combining
            first = None
            continue

        categories[cp] = category
        ccc[cp] = combining
        if decomposition and not decomposition.startswith("<"):
            decompositions[cp] = tuple(int(value, 16)
                                       for value in decomposition.split())

    if first is not None:
        raise ValueError("unterminated UnicodeData First record")
    return categories, ccc, decompositions


def parse_property_ranges(text: str, property_name: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for raw in text.splitlines():
        record = raw.split("#", 1)[0].strip()
        if not record:
            continue
        fields = [field.strip() for field in record.split(";")]
        if len(fields) < 2 or fields[1] != property_name:
            continue
        endpoints = fields[0].split("..")
        lo = int(endpoints[0], 16)
        hi = int(endpoints[-1], 16)
        result.append((lo, hi))
    return result


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if result and lo <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def nonzero_runs(values: list[int]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    start: int | None = None
    previous = 0
    for cp in range(MAX_CODEPOINT + 1):
        value = values[cp] if cp < MAX_CODEPOINT else 0
        if value == previous:
            continue
        if previous:
            assert start is not None
            result.append((start, cp - 1, previous))
        start = cp if value else None
        previous = value
    return result


def refresh_cache(ucd9_path: Path, ucd16_path: Path) -> bytes:
    if sha256(ucd9_path) != UCD9_SHA256:
        raise ValueError(f"unexpected SHA-256 for {ucd9_path}")
    if sha256(ucd16_path) != UCD16_SHA256:
        raise ValueError(f"unexpected SHA-256 for {ucd16_path}")

    with zipfile.ZipFile(ucd16_path) as ucd16:
        categories16, _, _ = parse_unicode_data(
            zip_text(ucd16, "UnicodeData.txt"))
        whitespace = merge_ranges(parse_property_ranges(
            zip_text(ucd16, "PropList.txt"), "White_Space"))
    with zipfile.ZipFile(ucd9_path) as ucd9:
        _, ccc9, decompositions9 = parse_unicode_data(
            zip_text(ucd9, "UnicodeData.txt"))
        exclusions9 = parse_property_ranges(
            zip_text(ucd9, "DerivedNormalizationProps.txt"),
            "Full_Composition_Exclusion")

    category_values = [CLASS_OTHER] * MAX_CODEPOINT
    for cp, category in enumerate(categories16):
        if category.startswith("L"):
            category_values[cp] = CLASS_LETTER
        elif category.startswith("M"):
            category_values[cp] = CLASS_MARK
        elif category.startswith("N"):
            category_values[cp] = CLASS_NUMBER

    excluded: set[int] = set()
    for lo, hi in exclusions9:
        excluded.update(range(lo, hi + 1))

    lines = [
        "# ds4 Qwen Unicode semantic cache v1",
        "# Derived from official Unicode Character Database archives.",
        f"# UCD 9.0.0 UCD.zip SHA-256: {UCD9_SHA256}",
        f"# UCD 16.0.0 UCD.zip SHA-256: {UCD16_SHA256}",
        "# License: tests/qwen/UNICODE_DATA_LICENSE.txt",
        "# C lo hi class: UCD16 L=1 M=2 N=3; zero ranges are implicit",
        "# W lo hi: UCD16 White_Space",
        "# K lo hi ccc: UCD9 nonzero canonical combining class",
        "# D cp sequence...: UCD9 raw canonical decomposition",
        "# X cp: UCD9 two-codepoint Full_Composition_Exclusion",
    ]
    for lo, hi, value in nonzero_runs(category_values):
        lines.append(f"C {lo:06X} {hi:06X} {value}")
    for lo, hi in whitespace:
        lines.append(f"W {lo:06X} {hi:06X}")
    for lo, hi, value in nonzero_runs(ccc9):
        lines.append(f"K {lo:06X} {hi:06X} {value}")
    for cp, mapping in sorted(decompositions9.items()):
        mapped = " ".join(f"{value:06X}" for value in mapping)
        lines.append(f"D {cp:06X} {mapped}")
    for cp, mapping in sorted(decompositions9.items()):
        if len(mapping) == 2 and cp in excluded:
            lines.append(f"X {cp:06X}")
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_cache(data: bytes) -> SourceData:
    decoded = data.decode("ascii")
    for required in (
            f"# UCD 9.0.0 UCD.zip SHA-256: {UCD9_SHA256}",
            f"# UCD 16.0.0 UCD.zip SHA-256: {UCD16_SHA256}"):
        if required not in decoded:
            raise ValueError(f"semantic cache is missing provenance: {required}")
    categories: list[tuple[int, int, int]] = []
    whitespace: list[tuple[int, int]] = []
    ccc: list[tuple[int, int, int]] = []
    decompositions: list[tuple[int, tuple[int, ...]]] = []
    exclusions: set[int] = set()

    for line_number, raw in enumerate(decoded.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        kind = fields[0]
        try:
            if kind == "C" and len(fields) == 4:
                categories.append((int(fields[1], 16), int(fields[2], 16),
                                   int(fields[3])))
            elif kind == "W" and len(fields) == 3:
                whitespace.append((int(fields[1], 16), int(fields[2], 16)))
            elif kind == "K" and len(fields) == 4:
                ccc.append((int(fields[1], 16), int(fields[2], 16),
                            int(fields[3])))
            elif kind == "D" and len(fields) in (3, 4):
                decompositions.append((int(fields[1], 16),
                                       tuple(int(value, 16)
                                             for value in fields[2:])))
            elif kind == "X" and len(fields) == 2:
                exclusions.add(int(fields[1], 16))
            else:
                raise ValueError("unknown record shape")
        except ValueError as error:
            raise ValueError(f"cache line {line_number}: {raw!r}: {error}") from error

    source = SourceData(tuple(categories), tuple(whitespace), tuple(ccc),
                        tuple(decompositions), frozenset(exclusions))
    validate_source(source)
    return source


def validate_ranges(ranges: tuple[tuple[int, int, int], ...], name: str,
                    max_value: int) -> None:
    previous_end = -1
    for lo, hi, value in ranges:
        if not (previous_end < lo <= hi < MAX_CODEPOINT):
            raise ValueError(f"invalid or overlapping {name} range {lo:X}..{hi:X}")
        if not (1 <= value <= max_value):
            raise ValueError(f"invalid {name} value {value}")
        previous_end = hi


def validate_source(source: SourceData) -> None:
    validate_ranges(source.category_ranges, "category", CLASS_NUMBER)
    validate_ranges(source.ccc_ranges, "CCC", 255)
    if len(source.category_ranges) != 1142:
        raise ValueError("unexpected UCD16 category range count")
    if len(source.whitespace_ranges) != 10:
        raise ValueError("unexpected UCD16 White_Space range count")
    previous_end = -1
    for lo, hi in source.whitespace_ranges:
        if not (previous_end < lo <= hi < MAX_CODEPOINT):
            raise ValueError("invalid or overlapping White_Space ranges")
        previous_end = hi
    if len(source.ccc_ranges) != 333:
        raise ValueError("unexpected UCD9 CCC range count")
    if len(source.decompositions) != 2060:
        raise ValueError("unexpected UCD9 canonical decomposition count")
    if len(source.composition_exclusions) != 85:
        raise ValueError("unexpected relevant composition exclusion count")
    previous = -1
    decomposition_cps = set()
    for cp, mapping in source.decompositions:
        if cp <= previous or cp >= MAX_CODEPOINT or len(mapping) not in (1, 2):
            raise ValueError("invalid canonical decomposition table")
        if any(value >= MAX_CODEPOINT for value in mapping):
            raise ValueError("invalid decomposition scalar")
        previous = cp
        decomposition_cps.add(cp)
    if not source.composition_exclusions <= decomposition_cps:
        raise ValueError("composition exclusion without canonical decomposition")
    decomposition_map = dict(source.decompositions)
    if any(len(decomposition_map[cp]) != 2
           for cp in source.composition_exclusions):
        raise ValueError("irrelevant composition exclusion in semantic cache")


def ranges_to_boundaries(
        ranges: tuple[tuple[int, int, int], ...]) -> list[tuple[int, int]]:
    events: dict[int, int] = {0: 0}
    for lo, hi, value in ranges:
        events[lo] = value
        events[hi + 1] = 0
    result: list[tuple[int, int]] = []
    current = None
    for cp, value in sorted(events.items()):
        if value == current:
            continue
        result.append((cp, value))
        current = value
    return result


def ccc_lookup(ranges: tuple[tuple[int, int, int], ...], cp: int) -> int:
    lo = 0
    hi = len(ranges)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if ranges[mid][0] <= cp:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return 0
    start, end, value = ranges[lo - 1]
    return value if cp <= end else 0


def generate(source: SourceData, cache_digest: str) -> bytes:
    category_boundaries = ranges_to_boundaries(source.category_ranges)
    ccc_boundaries = ranges_to_boundaries(source.ccc_ranges)
    raw_decompositions = dict(source.decompositions)

    expanded_cache: dict[int, tuple[int, ...]] = {}
    active: set[int] = set()

    def expand(cp: int) -> tuple[int, ...]:
        if cp in expanded_cache:
            return expanded_cache[cp]
        mapping = raw_decompositions.get(cp)
        if mapping is None:
            return (cp,)
        if cp in active:
            raise ValueError(f"canonical decomposition cycle at U+{cp:04X}")
        active.add(cp)
        result = tuple(value for part in mapping for value in expand(part))
        active.remove(cp)
        expanded_cache[cp] = result
        return result

    decomposition_cps: list[int] = []
    decomposition_meta: list[int] = []
    decomposition_pool: list[int] = []
    for cp, _ in source.decompositions:
        mapping = expand(cp)
        if not 1 <= len(mapping) <= 4:
            raise ValueError(f"unexpected expanded decomposition length for U+{cp:04X}")
        offset = len(decomposition_pool)
        if offset >= 1 << 12:
            raise ValueError("decomposition pool no longer fits 12-bit offsets")
        decomposition_cps.append(cp)
        decomposition_meta.append(offset | ((len(mapping) - 1) << 12))
        decomposition_pool.extend(mapping)

    composition: list[tuple[int, int, int]] = []
    for cp, mapping in source.decompositions:
        if (len(mapping) == 2 and cp not in source.composition_exclusions and
                ccc_lookup(source.ccc_ranges, mapping[0]) == 0):
            composition.append((mapping[0], mapping[1], cp))
    composition.sort(key=lambda item: (item[0], item[1]))
    composition_packed = [first | (second << 21) | (result << 42)
                          for first, second, result in composition]

    if len(category_boundaries) != 1962 or len(ccc_boundaries) != 496:
        raise ValueError("unexpected generated boundary count")
    if len(decomposition_pool) != 3404 or len(composition) != 940:
        raise ValueError("unexpected generated NFC table count")

    lines = [
        "/* Generated by tests/gen_qwen_unicode.py; do not edit.",
        " * Qwen3.6 tokenizer semantics:",
        " *   - General_Category and White_Space: Unicode 16.0.0",
        " *   - NFC normalization: Unicode 9.0.0",
        f" * Semantic cache SHA-256: {cache_digest}",
        " * Data license: tests/qwen/UNICODE_DATA_LICENSE.txt",
        " */",
        "",
    ]

    def emit_array(c_type: str, name: str, values: list[int], width: int,
                   per_line: int, suffix: str = "") -> None:
        lines.append(f"static const {c_type} {name}[{len(values)}] = {{")
        for offset in range(0, len(values), per_line):
            part = values[offset:offset + per_line]
            rendered = ", ".join(f"0x{value:0{width}X}{suffix}"
                                   for value in part)
            lines.append(f"    {rendered},")
        lines.append("};")
        lines.append("")

    emit_array("uint32_t", "qwen_uc_category_boundaries",
               [(cp << 2) | value for cp, value in category_boundaries],
               8, 8)
    space_values = [value for lo, hi in source.whitespace_ranges
                    for value in (lo, hi)]
    emit_array("uint32_t", "qwen_uc_space_ranges", space_values, 8, 8)
    emit_array("uint32_t", "qwen_uc_ccc_boundaries",
               [(cp << 8) | value for cp, value in ccc_boundaries], 8, 8)
    emit_array("uint32_t", "qwen_uc_decomposition_cps",
               decomposition_cps, 8, 8)
    emit_array("uint16_t", "qwen_uc_decomposition_meta",
               decomposition_meta, 4, 12)
    emit_array("uint32_t", "qwen_uc_decomposition_pool",
               decomposition_pool, 8, 8)
    emit_array("uint64_t", "qwen_uc_composition",
               composition_packed, 16, 4, "ULL")
    return ("\n".join(lines) + "\n").encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and verify the committed include")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="rebuild the semantic cache from pinned official UCD zips")
    parser.add_argument("--ucd9", type=Path,
                        help="official Unicode 9.0.0 UCD.zip")
    parser.add_argument("--ucd16", type=Path,
                        help="official Unicode 16.0.0 UCD.zip")
    args = parser.parse_args()

    if args.check and args.refresh_cache:
        parser.error("--check and --refresh-cache are mutually exclusive")
    if args.refresh_cache:
        if args.ucd9 is None or args.ucd16 is None:
            parser.error("--refresh-cache requires --ucd9 and --ucd16")
        cache_bytes = refresh_cache(args.ucd9, args.ucd16)
        refreshed_digest = hashlib.sha256(cache_bytes).hexdigest()
        if refreshed_digest != EXPECTED_CACHE_SHA256:
            print(f"error: refreshed semantic cache SHA-256 is "
                  f"{refreshed_digest}, expected {EXPECTED_CACHE_SHA256}",
                  file=sys.stderr)
            return 1
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_bytes(cache_bytes)

    try:
        cache_bytes = args.cache.read_bytes()
    except OSError as error:
        print(f"error: cannot read semantic cache: {error}", file=sys.stderr)
        return 1
    cache_digest = hashlib.sha256(cache_bytes).hexdigest()
    if cache_digest != EXPECTED_CACHE_SHA256:
        print(f"error: semantic cache SHA-256 is {cache_digest}, expected "
              f"{EXPECTED_CACHE_SHA256}", file=sys.stderr)
        return 1
    source = parse_cache(cache_bytes)
    expected = generate(source, cache_digest)

    if args.check:
        try:
            actual = args.output.read_bytes()
        except OSError as error:
            print(f"error: cannot read generated include: {error}", file=sys.stderr)
            return 1
        if actual != expected:
            print(f"error: {args.output} is stale; run {Path(__file__).name}",
                  file=sys.stderr)
            return 1
        print(f"Qwen Unicode tables: OK ({len(actual)} source bytes)")
        return 0

    args.output.write_bytes(expected)
    print(f"wrote {args.output} ({len(expected)} bytes, "
          f"SHA-256 {hashlib.sha256(expected).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
