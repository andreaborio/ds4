#!/usr/bin/env python3
"""Unit tests for release metadata/version binding."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("test_release_metadata.py")
SPEC = importlib.util.spec_from_file_location("release_metadata", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
RELEASE_METADATA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RELEASE_METADATA
SPEC.loader.exec_module(RELEASE_METADATA)


class ReleaseMetadataUnitTests(unittest.TestCase):
    def test_expected_version_matches_or_is_optional(self) -> None:
        RELEASE_METADATA.require_expected_version("0.3.0", None)
        RELEASE_METADATA.require_expected_version("0.3.0", "")
        RELEASE_METADATA.require_expected_version("0.3.0", "0.3.0")

    def test_expected_version_mismatch_and_invalid_value_fail(self) -> None:
        with self.assertRaisesRegex(SystemExit, "does not match"):
            RELEASE_METADATA.require_expected_version("0.3.0", "9.9.9")
        with self.assertRaisesRegex(SystemExit, "invalid RELEASE_VERSION"):
            RELEASE_METADATA.require_expected_version("0.3.0", "v0.3.0")


if __name__ == "__main__":
    unittest.main()
