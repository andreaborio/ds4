#!/usr/bin/env python3
"""Collect and verify the pinned Qwen3.8-Flash-Next source inventory.

This is Phase 0 provenance evidence for the Qwen4Exp work package.  It reads
only immutable published metadata: the Hugging Face blob list for one pinned
revision and the safetensors headers reached by bounded HTTP range requests.
No tensor payload is materialized and no model runs.

The emitted fixture records every file digest, every shard digest, and every
tensor name, dtype, shape, owning shard, and byte extent.  Re-running with
``--check`` re-derives the fixture from the network and fails on any
difference, so a later reviewer can reproduce the same inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "qwen4exp" / "fixtures" / "qwen38flash-next-inventory-v1.json"

REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"

# Files whose digests are pinned from a downloaded copy at the pinned revision.
# Non-LFS repository files do not carry an LFS sha256 in the blob API, so the
# only reproducible digest for them is the SHA-256 of the served bytes.
DOWNLOAD_SHA256_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

# The mathematical reference lives in the Transformers package, not in the
# model repository: the pinned config declares no auto-map, so these two files
# at the pinned Transformers commit define the executable semantics.
TRANSFORMERS_SOURCES = (
    "src/transformers/models/qwen4_exp/configuration_qwen4_exp.py",
    "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py",
)
TRANSFORMERS_RAW = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_COMMIT}/"
)

# Byte buffers needed to prove the PLE hash constants are exactly reproducible
# from the pinned algorithm without opening a 360 GB checkpoint.
PLE_HASH_BUFFERS = (
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
)

EXCLUDED_PREFIXES = {
    "base": "model.language_model.",
    "mtp": "mtp.",
    "vision": "model.visual.",
}

# bos_token_id, eos_token_id and pad_token_id are all 248044 in the pinned
# config.json; it is also the id the PLE n-gram history treats as a sentinel.
PADDING_TOKEN = 248044

TIMEOUT = 120


def fetch(url: str, start: int | None = None, end: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={})
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = response.read()
    if start is not None and len(body) != end - start:
        raise RuntimeError(f"short range read {url}: {len(body)} of {end - start}")
    return body


def blob_list() -> list[dict]:
    url = (
        "https://huggingface.co/api/models/"
        f"{REPOSITORY}?blobs=true&revision={REVISION}"
    )
    payload = json.load(urllib.request.urlopen(url, timeout=TIMEOUT))
    siblings = payload["siblings"]
    files = []
    for entry in siblings:
        lfs = entry.get("lfs") or {}
        files.append(
            {
                "path": entry["rfilename"],
                "size": lfs.get("size", entry.get("size")),
                "sha256": lfs.get("sha256"),
                "lfs": bool(lfs),
            }
        )
    return sorted(files, key=lambda row: row["path"])


def download_digest(path: str) -> str:
    url = (
        f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{path}"
    )
    return hashlib.sha256(fetch(url)).hexdigest()


def safetensors_header(shard: str) -> tuple[dict, int]:
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{shard}?blob"
    prefix = fetch(url, 0, 8)
    header_len = struct.unpack("<Q", prefix)[0]
    header = json.loads(fetch(url, 8, 8 + header_len))
    return header, 8 + header_len


def int_buffer(shard: str, base: int, entry: dict) -> list[int]:
    begin, end = entry["data_offsets"]
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{shard}?blob"
    raw = fetch(url, base + begin, base + end)
    count = len(raw) // 8
    return list(struct.unpack("<" + "q" * count, raw))


def row_is_zero(shard: str, base: int, entry: dict, row: int, row_bytes: int) -> bool:
    """Check one 2-byte-per-element row without materializing the tensor.

    ``entry`` is an inventory tensor record built by :func:`collect`, so its
    ``begin`` is already relative to the shard data area that ``base`` locates.
    """

    start = base + entry["begin"] + row * row_bytes
    raw = fetch(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{shard}?blob",
                start, start + row_bytes)
    return len(raw) == row_bytes and raw.count(0) == row_bytes


def padding_row_facts(tensors: list[dict], bases: dict[str, int]) -> dict:
    """Probe the padding row of the two 248,320 x 2,560 token tables.

    ``Qwen4ExpTextModel`` builds ``embed_tokens`` with ``padding_idx`` set to the
    id shared by bos, eos and pad, so a published checkpoint is expected to keep
    that row exactly zero.  The probe reads 5 KiB per table instead of opening a
    1.2 GiB tensor.
    """

    row_bytes = 2560 * 2
    facts = {
        "token": PADDING_TOKEN,
        "rowBytes": row_bytes,
        "checked": [],
    }
    for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        entry = next(t for t in tensors if t["name"] == name)
        if entry["shape"] != [248320, 2560] or entry["dtype"] != "BF16":
            raise RuntimeError(f"{name}: unexpected padding probe shape")
        facts["checked"].append(
            {
                "name": name,
                "shard": entry["shard"],
                "allZero": row_is_zero(
                    entry["shard"], bases[entry["shard"]], entry, PADDING_TOKEN, row_bytes
                ),
            }
        )
    facts["embedZero"] = next(
        row["allZero"] for row in facts["checked"]
        if row["name"].startswith("model.language_model.")
    )
    return facts


def collect() -> dict:
    files = blob_list()
    by_path = {row["path"]: row for row in files}
    for path in DOWNLOAD_SHA256_FILES:
        row = by_path[path]
        row["sha256"] = download_digest(path)
        row["lfs"] = False
        row["digestSource"] = "pinned-download"
    for row in files:
        if row["lfs"]:
            row["digestSource"] = "api-lfs"
        else:
            row.setdefault("digestSource", "git-blob")

    shards = [
        row for row in files
        if row["path"].startswith("model-") and row["path"].endswith(".safetensors")
    ]
    if len(shards) != 131:
        raise RuntimeError(f"expected 131 shards, found {len(shards)}")

    index = json.loads(
        fetch(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
              "model.safetensors.index.json")
    )
    weight_map = index["weight_map"]
    tensors: list[dict] = []
    ple_buffers: dict[str, list[int]] = {}
    bases: dict[str, int] = {}
    for shard_row in shards:
        header, base = safetensors_header(shard_row["path"])
        bases[shard_row["path"]] = base
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            tensors.append(
                {
                    "name": name,
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "shard": shard_row["path"],
                    "begin": begin,
                    "end": end,
                }
            )
            if name in PLE_HASH_BUFFERS:
                ple_buffers[name] = int_buffer(shard_row["path"], base, entry)
    tensors.sort(key=lambda row: row["name"])
    names = {row["name"] for row in tensors}
    if names != set(weight_map):
        raise RuntimeError("index weight_map and shard headers disagree")
    if len(tensors) != len(weight_map):
        raise RuntimeError("tensor count does not match the index")

    for name, shard in weight_map.items():
        row = next(t for t in tensors if t["name"] == name)
        if row["shard"] != shard:
            raise RuntimeError(f"{name}: index shard {shard} != header {row['shard']}")

    canonical = b""
    for row in tensors:
        canonical += "\0".join(
            (
                row["name"],
                row["dtype"],
                ",".join(str(v) for v in row["shape"]),
                row["shard"],
                str(row["begin"]),
                str(row["end"]),
            )
        ).encode("utf-8") + b"\n"

    bytes_by_dtype: dict[str, int] = {}
    classification: dict[str, int] = {}
    for row in tensors:
        span = row["end"] - row["begin"]
        bytes_by_dtype[row["dtype"]] = bytes_by_dtype.get(row["dtype"], 0) + span
        group = "base"
        for label, prefix in EXCLUDED_PREFIXES.items():
            if row["name"].startswith(prefix) and label != "base":
                group = label
        if row["name"].startswith("model.visual."):
            group = "vision"
        classification[group] = classification.get(group, 0) + 1
        if row["name"].startswith("model.language_model.layers.1.ple."):
            classification["ple"] = classification.get("ple", 0) + 1

    ngram_rows = sum(
        row["shape"][0] for row in tensors if ".ngram_embedding.shard_" in row["name"]
    )
    ngram_cols = {row["shape"][1] for row in tensors if ".ngram_embedding.shard_" in row["name"]}

    padding = padding_row_facts(tensors, bases)

    transformers_sources = []
    for path in TRANSFORMERS_SOURCES:
        body = fetch(TRANSFORMERS_RAW + path)
        transformers_sources.append(
            {
                "path": path,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )

    return {
        "schemaVersion": 1,
        "kind": "qwen38flash-next-source-inventory",
        "repository": REPOSITORY,
        "revision": REVISION,
        "transformersCommit": TRANSFORMERS_COMMIT,
        "files": files,
        "transformersSources": transformers_sources,
        "shardCount": len(shards),
        "shardSha256": {row["path"]: row["sha256"] for row in shards},
        "totalBytes": index["metadata"]["total_size"],
        "tensorCount": len(tensors),
        "tensorInventorySha256": hashlib.sha256(canonical).hexdigest(),
        "bytesByDtype": dict(sorted(bytes_by_dtype.items())),
        "classification": dict(sorted(classification.items())),
        "ngramEmbeddingRows": ngram_rows,
        "ngramEmbeddingCols": sorted(ngram_cols),
        "paddingRowFacts": padding,
        "pleHashBuffers": {k: v for k, v in sorted(ple_buffers.items())},
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the fixture")
    parser.add_argument("--check", action="store_true", help="verify the fixture")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("use exactly one of --write or --check")

    document = collect()
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(encoded)
        print(f"wrote {OUT.relative_to(ROOT)}")
        return 0

    current = OUT.read_text()
    if current != encoded:
        print(
            "fixture drift: the pinned inventory no longer matches the "
            "re-derived source",
            file=sys.stderr,
        )
        return 1
    print(f"ok: {OUT.relative_to(ROOT)} matches the pinned source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
