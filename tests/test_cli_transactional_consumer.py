#!/usr/bin/env python3
"""Build and run the model-free transactional CLI consumer harness."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "internal" / "test_cli_transactional_consumer.c"


def main() -> int:
    compiler = os.environ.get("CC", "cc")
    with tempfile.TemporaryDirectory(prefix="ds4-cli-transaction-") as tmp:
        binary = Path(tmp) / "test_cli_transactional_consumer"
        linker_gc = "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
        compile_run = subprocess.run(
            [
                compiler,
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                # This harness includes the real CLI translation unit so its
                # static frontend functions are callable. Other executable
                # modes are intentionally unreachable and dead-stripped.
                "-Wno-unused-function",
                "-ffunction-sections",
                linker_gc,
                "-I",
                str(ROOT),
                str(SOURCE),
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if compile_run.returncode != 0:
            print(compile_run.stdout, end="")
            print(compile_run.stderr, end="", file=os.sys.stderr)
            return compile_run.returncode
        test_run = subprocess.run(
            [str(binary)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        print(test_run.stdout, end="")
        print(test_run.stderr, end="", file=os.sys.stderr)
        return test_run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
