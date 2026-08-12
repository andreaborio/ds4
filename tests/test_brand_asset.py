#!/usr/bin/env python3
"""Freeze the Hebrus logo and validate its public repository artwork."""

from __future__ import annotations

import hashlib
import pathlib
import struct


ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSET = ROOT / "docs" / "media" / "hebrus-logo.png"
SOCIAL_PREVIEW = ROOT / "docs" / "media" / "hebrus-social-preview.png"
SOCIAL_PREVIEW_SOURCE = ROOT / "docs" / "media" / "hebrus-social-preview.svg"
README = ROOT / "README.md"
EXPECTED_SHA256 = "4be8949c73bd52e7abef58396dcd57f636165a8bb6cd6d536a600bcbf880594c"
EXPECTED_SIZE = (1254, 1254)
EXPECTED_SOCIAL_PREVIEW_SIZE = (1280, 640)


def png_header(path: pathlib.Path) -> tuple[int, int, int, int]:
    payload = path.read_bytes()
    if payload[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise AssertionError(
            f"{path.relative_to(ROOT)} is not a PNG with an IHDR first chunk"
        )
    return struct.unpack(">IIBB", payload[16:26])


def main() -> None:
    payload = ASSET.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(
            f"{ASSET.relative_to(ROOT)} was modified: expected {EXPECTED_SHA256}, got {digest}"
        )

    width, height, bit_depth, color_type = png_header(ASSET)
    if (width, height) != EXPECTED_SIZE:
        raise AssertionError(f"unexpected logo dimensions: {width}x{height}")
    if (bit_depth, color_type) != (8, 6):
        raise AssertionError("Hebrus logo must retain its original 8-bit RGBA representation")

    if "docs/media/hebrus-logo.png" not in README.read_text(encoding="utf-8"):
        raise AssertionError("README does not use the canonical Hebrus logo")

    social_width, social_height, _, _ = png_header(SOCIAL_PREVIEW)
    if (social_width, social_height) != EXPECTED_SOCIAL_PREVIEW_SIZE:
        raise AssertionError(
            f"unexpected social-preview dimensions: {social_width}x{social_height}"
        )

    social_source = SOCIAL_PREVIEW_SOURCE.read_text(encoding="utf-8")
    for token in (
        "Hebrus",
        "Metal-first MoE inference",
        "Apple Silicon",
        "hebrus-logo.png",
    ):
        if token not in social_source:
            raise AssertionError(f"social-preview source is missing {token!r}")
    if "Hebrus Studio" in social_source:
        raise AssertionError("engine social preview must not use Hebrus Studio branding")

    print(
        f"brand-asset: PASS (logo {width}x{height}, RGBA, {digest}; "
        f"social preview {social_width}x{social_height})"
    )


if __name__ == "__main__":
    main()
