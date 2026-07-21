#!/usr/bin/env python3
"""Model-backed Hebrus/DS4 server route, output, and shutdown parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN_SIZE = 20_808_566_880
QWEN_SHA256 = "dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise AssertionError(f"{method} {url} returned HTTP {response.status}")
        return json.loads(response.read())


def normalize(document: Any) -> Any:
    if isinstance(document, dict):
        return {
            key: normalize(value)
            for key, value in sorted(document.items())
            if key not in {"id", "created"}
        }
    if isinstance(document, list):
        return [normalize(value) for value in document]
    return document


def wait_for_models(base_url: str, process: subprocess.Popen[bytes], timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "server did not accept a request"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited during startup with code {process.returncode}")
        try:
            return http_json(f"{base_url}/v1/models")
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise AssertionError(f"server startup timed out: {last_error}")


def run_server(
    binary: pathlib.Path,
    model: pathlib.Path,
    port: int,
    startup_timeout: int,
    evidence_dir: pathlib.Path,
) -> dict[str, Any]:
    log_path = evidence_dir / f"{binary.name}.log"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DS4_", "HEBRUS_"))
    }
    command = [
        str(binary),
        "-m", str(model),
        "--ctx", "8192",
        "--tokens", "32",
        "--threads", "8",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            models = wait_for_models(base_url, process, startup_timeout)
            entries = models.get("data")
            if not isinstance(entries, list) or len(entries) != 1:
                raise AssertionError(f"{binary.name}: unexpected model list: {models}")
            model_id = entries[0].get("id")
            if not isinstance(model_id, str) or not model_id:
                raise AssertionError(f"{binary.name}: model list has no usable id")
            model_document = http_json(
                f"{base_url}/v1/models/{urllib.parse.quote(model_id, safe='')}"
            )
            completion = http_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with exactly: HEBRUS ALIAS PARITY OK",
                        }
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                    "seed": 20260721,
                    "thinking": False,
                    "stream": False,
                },
            )
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    raise AssertionError(f"{binary.name}: graceful shutdown timed out")
    if process.returncode != 0:
        raise AssertionError(f"{binary.name}: shutdown returned {process.returncode}; see {log_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in ("listening on http://127.0.0.1", "shutdown requested, draining requests"):
        if marker not in log_text:
            raise AssertionError(f"{binary.name}: missing log marker {marker!r}")
    return {
        "command": command,
        "models": normalize(models),
        "model": normalize(model_document),
        "completion": normalize(completion),
        "log": str(log_path),
        "exit_code": process.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--bin-dir", default=ROOT, type=pathlib.Path)
    parser.add_argument("--port", default=18080, type=int)
    parser.add_argument("--startup-timeout", default=180, type=int)
    parser.add_argument("--evidence-dir", type=pathlib.Path)
    parser.add_argument(
        "--skip-model-hash",
        action="store_true",
        help="skip the full GGUF digest only when it was verified separately in this cohort",
    )
    args = parser.parse_args()

    model = args.model.expanduser().resolve(strict=True)
    if model.stat().st_size != QWEN_SIZE:
        raise AssertionError(f"unexpected Qwen size: {model.stat().st_size}")
    model_hash = "not-computed"
    if not args.skip_model_hash:
        model_hash = sha256(model)
        if model_hash != QWEN_SHA256:
            raise AssertionError(f"unexpected Qwen SHA-256: {model_hash}")

    bin_dir = args.bin_dir.resolve(strict=True)
    canonical = bin_dir / "hebrus-server"
    legacy = bin_dir / "ds4-server"
    if not canonical.exists() or not legacy.exists() or not os.path.samefile(canonical, legacy):
        raise AssertionError("hebrus-server and ds4-server are not one executable surface")

    if args.port < 1024 or args.port > 65535:
        raise AssertionError("--port must be an unprivileged TCP port")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.evidence_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="hebrus-server-alias-")
        evidence_dir = pathlib.Path(temporary.name)
    else:
        evidence_dir = args.evidence_dir.resolve()
        evidence_dir.mkdir(parents=True, exist_ok=False)

    try:
        results = {
            canonical.name: run_server(
                canonical, model, args.port, args.startup_timeout, evidence_dir
            ),
            legacy.name: run_server(
                legacy, model, args.port, args.startup_timeout, evidence_dir
            ),
        }
        for field in ("models", "model", "completion", "exit_code"):
            if results[canonical.name][field] != results[legacy.name][field]:
                raise AssertionError(f"server aliases differ for {field}; see {evidence_dir}")

        report = {
            "schema_version": 1,
            "model": {
                "path": str(model),
                "bytes": model.stat().st_size,
                "sha256": model_hash,
            },
            "binary_sha256": sha256(canonical),
            "port": args.port,
            "results": results,
        }
        report_path = evidence_dir / "server-alias-parity.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"server-alias-model: PASS ({canonical.name}/{legacy.name}, {report_path})")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"server-alias-model: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
