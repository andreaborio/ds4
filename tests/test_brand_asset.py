#!/usr/bin/env python3
"""Freeze the maintainer-supplied Hebrus logo and its README use."""

from __future__ import annotations

import hashlib
import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSET = ROOT / "docs" / "media" / "hebrus-logo.png"
README = ROOT / "README.md"
EXPECTED_SHA256 = "4be8949c73bd52e7abef58396dcd57f636165a8bb6cd6d536a600bcbf880594c"
EXPECTED_SIZE = (1254, 1254)


def main() -> None:
    payload = ASSET.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(
            f"{ASSET.relative_to(ROOT)} was modified: expected {EXPECTED_SHA256}, got {digest}"
        )

    if payload[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise AssertionError("Hebrus logo is not a canonical PNG with an IHDR first chunk")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[16:26])
    if (width, height) != EXPECTED_SIZE:
        raise AssertionError(f"unexpected logo dimensions: {width}x{height}")
    if (bit_depth, color_type) != (8, 6):
        raise AssertionError("Hebrus logo must retain its original 8-bit RGBA representation")

    if "docs/media/hebrus-logo.png" not in README.read_text(encoding="utf-8"):
        raise AssertionError("README does not use the canonical Hebrus logo")

    print(f"brand-asset: PASS ({width}x{height}, RGBA, {digest})")


if __name__ == "__main__":
    main()
