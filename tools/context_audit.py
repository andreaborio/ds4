#!/usr/bin/env python3
"""Enforce monotonic context hygiene for the agent-facing source tree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = ROOT / "tools" / "context_budget.json"
PORTABLE_HOME_PREFIXES = (
    "~/.config/",
    "~/.ds4/",
    "~/.pi/",
    "~/bin/",
)
PERSONAL_HOME_RE = re.compile(r"(?<![A-Za-z0-9_])~/[^\s`'\")\]}>,;]+")
RETIRED_DISTRIBUTED_FLAG_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"--role|"
    r"--layers|"
    r"--listen|"
    r"--coordinator|"
    r"--dist-prefill-chunk|"
    r"--dist-prefill-window|"
    r"--dist-activation-bits|"
    r"--dist-replay-check|"
    r"--debug"
    r")(?![A-Za-z0-9_-])"
)


def tracked_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    items = tracked.stdout.split(b"\0") + untracked.stdout.split(b"\0")
    files = {ROOT / item.decode() for item in items if item}
    return sorted(path for path in files if path.exists())


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def contains_personal_path(line: str) -> bool:
    if "/Users/" in line:
        return True
    return any(
        not match.group(0).startswith(PORTABLE_HOME_PREFIXES)
        for match in PERSONAL_HOME_RE.finditer(line)
    )


def source_metrics(files: list[Path]) -> tuple[dict[str, int], dict[str, list[str]]]:
    root_markdown = 0
    implementation_includes: list[str] = []
    maybe_unused: list[str] = []
    env_names: set[str] = set()
    personal_paths: list[str] = []
    largest_lines = 0
    largest_file = ""

    # Implementation .inc partitions are source, not generated context hiding
    # places. Count their env reads, markers, and size like their including TU.
    source_suffixes = {".c", ".h", ".m", ".cu", ".cuh", ".metal", ".inc"}
    text_suffixes = source_suffixes | {".md", ".py", ".sh", ".json", ".txt"}
    include_re = re.compile(r'^\s*#\s*include\s+"\.\./[^\"]+\.c"', re.MULTILINE)
    env_re = re.compile(r'getenv\("(DS4_[A-Z0-9_]+)"\)')

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.parent == ROOT and path.suffix == ".md":
            root_markdown += 1
        if path.suffix not in text_suffixes:
            continue
        text = read_text(path)
        if path.suffix in source_suffixes:
            lines = text.count("\n") + (1 if text else 0)
            if lines > largest_lines:
                largest_lines = lines
                largest_file = rel
            env_names.update(env_re.findall(text))
            for line_no, line in enumerate(text.splitlines(), 1):
                if "MAYBE_UNUSED" in line and "#define" not in line:
                    maybe_unused.append(f"{rel}:{line_no}")
        if rel.startswith("tests/"):
            for match in include_re.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                implementation_includes.append(f"{rel}:{line_no}")
        for line_no, line in enumerate(text.splitlines(), 1):
            fixture = rel.startswith("tests/test-vectors/") or rel == "tests/long_context_security_prompt.txt"
            if contains_personal_path(line) and rel != "tools/context_audit.py" and not fixture:
                personal_paths.append(f"{rel}:{line_no}")

    metrics = {
        "root_markdown_files": root_markdown,
        "implementation_includes": len(implementation_includes),
        "maybe_unused_markers": len(maybe_unused),
        "direct_ds4_env_names": len(env_names),
        "personal_absolute_path_hits": len(personal_paths),
        "largest_source_lines": largest_lines,
    }
    details = {
        "direct_ds4_env_names": sorted(env_names),
        "implementation_includes": implementation_includes,
        "maybe_unused_markers": maybe_unused,
        "personal_absolute_path_hits": personal_paths,
        "largest_source": [f"{largest_file}:{largest_lines}"],
    }
    return metrics, details


def contract_issues(files: list[Path], allow_active_handoff: bool) -> list[str]:
    rels = {path.relative_to(ROOT).as_posix() for path in files}
    issues: list[str] = []

    required = {
        "AGENTS.md",
        "CONTRIBUTING.md",
        "QA_BEFORE_RELEASES.md",
        "docs/architecture/CODEMAP.md",
        "docs/contracts/RUNTIME_SUPPORT.md",
        "docs/work/HANDOFF_TEMPLATE.md",
    }
    for rel in sorted(required - rels):
        issues.append(f"missing required agent context: {rel}")
    if "AGENT.md" in rels:
        issues.append("AGENT.md is obsolete; keep the canonical instructions in AGENTS.md")

    agents = read_text(ROOT / "AGENTS.md")
    for canonical in ("CONTRIBUTING.md", "QA_BEFORE_RELEASES.md"):
        if agents and canonical not in agents:
            issues.append(f"AGENTS.md must route agents to {canonical}")

    frozen_paths = [
        rel
        for rel in rels
        if rel in {
            "ds4_cuda.cu",
            "ds4_iq2_tables_cuda.inc",
            "ds4_rocm.cu",
            "ds4_rocm.h",
            "tests/cuda_long_context_smoke.c",
        }
        or rel.startswith("rocm/")
    ]
    if frozen_paths:
        issues.append("frozen CUDA/ROCm sources remain tracked: " + ", ".join(sorted(frozen_paths)))

    generated_dataset = {
        "gguf-tools/imatrix/dataset/manifest.json",
        "gguf-tools/imatrix/dataset/prompts.jsonl",
        "gguf-tools/imatrix/dataset/rendered_prompts.txt",
        "gguf-tools/imatrix/dataset/rendered_prompts_nothink.txt",
        "gguf-tools/imatrix/dataset/rendered_prompts_think.txt",
    }
    tracked_generated_dataset = sorted(generated_dataset & rels)
    if tracked_generated_dataset:
        issues.append(
            "derived imatrix corpus must remain generated locally: "
            + ", ".join(tracked_generated_dataset)
        )

    active_source_suffixes = {".c", ".h", ".m", ".metal", ".cu", ".cuh", ".inc"}
    frozen_symbols = re.compile(r"DS4_BACKEND_CUDA|DS4_ROCM_BUILD|DS4_CUDA_")
    symbol_hits = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in active_source_suffixes or rel.startswith("gguf-tools/imatrix/"):
            continue
        if frozen_symbols.search(read_text(path)):
            symbol_hits.append(rel)
    if symbol_hits:
        issues.append("frozen CUDA/ROCm symbols remain in active source: " + ", ".join(symbol_hits))

    makefile = read_text(ROOT / "Makefile")
    frozen_targets = re.findall(
        r"(?m)^(cuda(?:-[A-Za-z0-9_.-]+)?|rocm(?:-[A-Za-z0-9_.-]+)?|strix-halo)\s*:",
        makefile,
    )
    if frozen_targets:
        issues.append("frozen CUDA/ROCm build targets returned: " + ", ".join(frozen_targets))

    retired_distributed_paths = {
        "ds4_distributed.c",
        "ds4_distributed.h",
    }
    present_distributed_paths = sorted(retired_distributed_paths & rels)
    if present_distributed_paths:
        issues.append(
            "retired distributed sources returned: "
            + ", ".join(present_distributed_paths)
        )

    retired_distributed_symbols = re.compile(
        r"\b(?:"
        r"ds4_dist_[A-Za-z0-9_]*|"
        r"ds4_distributed_(?:role|layers|options)|"
        r"DS4_DISTRIBUTED_[A-Za-z0-9_]*|"
        r"DS4_SESSION_LAYER_PAYLOAD_[A-Za-z0-9_]*|"
        r"ds4_engine_glm_layer_payload_bytes|"
        r"glm_layer_payload_tensor_bytes|"
        r"ds4_layer_payload_range_valid|"
        r"glm_graph_(?:alloc_slice|memory_guard_slice)|"
        r"load_(?:slice|layer_start|layer_end|output)|"
        r"ds4_session_(?:distributed_route_ready|slice_[A-Za-z0-9_]*|layer_slice_reset|"
        r"eval_layer_slice|eval_output_head_from_hc|layer_payload_bytes|"
        r"save_layer_payload|load_layer_payload)"
        r")\b"
    )
    distributed_symbol_hits = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in active_source_suffixes:
            continue
        if retired_distributed_symbols.search(read_text(path)):
            distributed_symbol_hits.append(rel)
    if distributed_symbol_hits:
        issues.append(
            "retired distributed symbols remain in active source: "
            + ", ".join(sorted(distributed_symbol_hits))
        )

    distributed_build_hits = []
    for rel in ("Makefile", "gguf-tools/Makefile"):
        text = read_text(ROOT / rel)
        if re.search(r"\bds4_distributed\.(?:c|h|o)\b", text):
            distributed_build_hits.append(rel)
    if distributed_build_hits:
        issues.append(
            "retired distributed build inputs remain: "
            + ", ".join(distributed_build_hits)
        )

    retired_flag_hits = []
    retired_flag_fixture = "tests/test_retired_distributed_flags.sh"
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix != ".sh" or rel == retired_flag_fixture:
            continue
        for line_no, line in enumerate(read_text(path).splitlines(), 1):
            flags = sorted(set(RETIRED_DISTRIBUTED_FLAG_RE.findall(line)))
            if flags:
                retired_flag_hits.append(f"{rel}:{line_no} ({', '.join(flags)})")
    if retired_flag_hits:
        issues.append(
            "retired distributed CLI flags remain in active scripts: "
            + ", ".join(retired_flag_hits)
        )

    if not allow_active_handoff:
        active = [
            rel
            for rel in rels
            if rel.startswith("docs/work/active/") and not rel.endswith("/.gitkeep")
        ]
        if active:
            issues.append("active handoffs must be resolved before merge: " + ", ".join(sorted(active)))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-active-handoff", action="store_true")
    args = parser.parse_args()

    files = tracked_files()
    metrics, details = source_metrics(files)
    issues = contract_issues(files, args.allow_active_handoff)

    if not args.report_only:
        if not args.budget.exists():
            issues.append(f"missing context budget: {args.budget}")
        else:
            budget = json.loads(args.budget.read_text(encoding="utf-8"))
            for name, policy in budget.items():
                actual = metrics.get(name)
                if actual is None:
                    issues.append(f"unknown context budget metric: {name}")
                    continue

                maximum = policy
                if isinstance(policy, dict):
                    maximum = policy.get("maximum")
                    baseline = policy.get("baseline")
                    reason = policy.get("reason")
                    classified = policy.get("accepted_additions")
                    if not isinstance(maximum, int):
                        issues.append(f"{name} context budget needs an integer maximum")
                        continue
                    if not isinstance(baseline, int):
                        issues.append(f"{name} classified context budget needs an integer baseline")
                    if not isinstance(reason, str) or not reason.strip():
                        issues.append(f"{name} classified context budget needs a reason")
                    if not isinstance(classified, dict) or not classified:
                        issues.append(f"{name} classified context budget needs accepted additions")
                    else:
                        additions: list[str] = []
                        for category, names in classified.items():
                            if not isinstance(category, str) or not isinstance(names, list) or not all(
                                isinstance(item, str) for item in names
                            ):
                                issues.append(f"{name} has an invalid accepted-addition category")
                                continue
                            additions.extend(names)
                        if len(additions) != len(set(additions)):
                            issues.append(f"{name} accepted additions contain duplicates")
                        if isinstance(baseline, int) and baseline + len(set(additions)) != maximum:
                            issues.append(
                                f"{name} classified budget does not match baseline plus accepted additions"
                            )
                        if name == "direct_ds4_env_names":
                            observed = set(details["direct_ds4_env_names"])
                            missing = sorted(set(additions) - observed)
                            if missing:
                                issues.append(
                                    f"{name} accepted additions are not direct source reads: "
                                    + ", ".join(missing)
                                )
                elif not isinstance(policy, int):
                    issues.append(f"{name} context budget must be an integer or policy object")
                    continue

                if actual > maximum:
                    issues.append(f"{name} increased: {actual} > budget {maximum}")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if issues:
        print("context audit failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        for name in ("implementation_includes", "maybe_unused_markers", "personal_absolute_path_hits"):
            entries = details[name]
            if entries:
                preview = ", ".join(entries[:8])
                suffix = " ..." if len(entries) > 8 else ""
                print(f"  {name}: {preview}{suffix}", file=sys.stderr)
        return 1
    print("context audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
