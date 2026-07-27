#!/usr/bin/env python3
"""Run the versioned Qwen 24 GiB request sequence against local Hebrus Studio."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_MANIFEST = FIXTURE_DIR / "qwen-24g-release-v1.json"


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def local_endpoint(base_url: str, endpoint: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("base URL must be an HTTP loopback address")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def parse_sse(response, evidence) -> tuple[str | None, int | None]:
    finish_reason: str | None = None
    completion_tokens: int | None = None
    saw_done = False
    for raw in response:
        evidence.write(raw)
        line = raw.decode("utf-8", errors="strict").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        document = json.loads(payload)
        usage = document.get("usage")
        if isinstance(usage, dict):
            value = usage.get("completion_tokens")
            if type(value) is int:
                completion_tokens = value
        choices = document.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                value = choice.get("finish_reason")
                if isinstance(value, str):
                    finish_reason = value
    if not saw_done:
        raise AssertionError("stream ended without data: [DONE]")
    return finish_reason, completion_tokens


def run_request(
    opener,
    url: str,
    headers: dict[str, str],
    manifest: dict,
    request_spec: dict,
    output_dir: Path,
    timeout: float,
) -> dict:
    prompt_spec = manifest["prompts"][request_spec["prompt"]]
    prompt_path = DEFAULT_MANIFEST.parent / prompt_spec["path"]
    prompt = prompt_path.read_text(encoding="utf-8")
    sampling = manifest["sampling"]
    body = {
        "model": manifest["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": manifest["stream"],
        "stream_options": {"include_usage": True},
        "reasoning_effort": request_spec["reasoningEffort"],
        "max_tokens": request_spec["maxTokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["topP"],
        "min_p": sampling["minP"],
        "seed": sampling["seed"],
    }
    request_id = request_spec["id"]
    started_at = timestamp()
    raw_path = output_dir / f"{request_id}.sse"
    http_request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with opener.open(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise AssertionError(f"{request_id}: HTTP {response.status}")
        with raw_path.open("wb") as evidence:
            finish_reason, completion_tokens = parse_sse(response, evidence)
    accepted = request_spec["acceptedFinishReasons"]
    if finish_reason not in accepted:
        raise AssertionError(
            f"{request_id}: finish_reason={finish_reason!r}, expected {accepted}"
        )
    minimum = request_spec["minimumGeneratedTokens"]
    if completion_tokens is None or completion_tokens < minimum:
        raise AssertionError(
            f"{request_id}: completion_tokens={completion_tokens!r}, "
            f"expected at least {minimum}"
        )
    return {
        "id": request_id,
        "startedAt": started_at,
        "finishedAt": timestamp(),
        "finishReason": finish_reason,
        "completionTokens": completion_tokens,
        "evidence": raw_path.name,
        "result": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:4242")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--api-key-env",
        default="HEBRUS_API_KEY",
        help="environment variable containing the optional Studio API key",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest.resolve() != DEFAULT_MANIFEST.resolve():
        raise SystemExit("only the checked-in qwen-24g-release-v1 manifest is accepted")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, mode=0o700)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    url = local_endpoint(args.base_url, manifest["endpoint"])
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    summary = {
        "schemaVersion": 1,
        "startedAt": timestamp(),
        "baseUrl": args.base_url,
        "manifest": str(args.manifest),
        "requests": [],
        "result": "RUNNING",
    }
    summary_path = args.output_dir / "summary.json"
    try:
        for request_spec in manifest["requests"]:
            result = run_request(
                opener,
                url,
                headers,
                manifest,
                request_spec,
                args.output_dir,
                args.timeout_seconds,
            )
            summary["requests"].append(result)
            print(
                f"{result['id']}: PASS "
                f"({result['completionTokens']} tokens, "
                f"finish={result['finishReason']})",
                flush=True,
            )
        summary["result"] = "PASS"
        return 0
    except (AssertionError, OSError, UnicodeError, urllib.error.URLError) as exc:
        summary["result"] = "FAIL"
        summary["error"] = str(exc)
        print(f"Qwen 24 GiB gate: FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        summary["finishedAt"] = timestamp()
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
