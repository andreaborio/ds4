#!/usr/bin/env python3
"""Validate the versioned Qwen 24 GiB physical release-request fixture."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIXTURE_DIR / "qwen-24g-release-v1.json"


def main() -> None:
    runner = Path(__file__).resolve().parent / "run_24g_release_gate.py"
    ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(document) == {
        "schemaVersion",
        "description",
        "endpoint",
        "model",
        "stream",
        "hostProfile",
        "sampling",
        "prompts",
        "requests",
    }
    assert document["schemaVersion"] == 1
    assert document["endpoint"] == "/v1/chat/completions"
    assert document["model"] == "qwen3.6-35b-a3b"
    assert document["stream"] is True
    assert document["hostProfile"] == {
        "physicalGiB": 24,
        "contextTokens": 16384,
        "maxOutputTokens": 8192,
    }
    assert document["sampling"] == {
        "temperature": 0,
        "topP": 1,
        "minP": 0,
        "seed": 20260727,
    }

    prompts = document["prompts"]
    assert set(prompts) == {"sarajevo", "sustained", "followup"}
    for name, prompt in prompts.items():
        assert set(prompt) == {"path", "sha256"}, name
        path = FIXTURE_DIR / prompt["path"]
        assert path.parent == FIXTURE_DIR
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == prompt["sha256"], name

    requests = document["requests"]
    assert [request["id"] for request in requests] == [
        "sarajevo-medium",
        "sustained-medium",
        "sarajevo-high",
        "sustained-high",
        "post-guard-followup",
    ]
    for request in requests:
        assert set(request) == {
            "id",
            "prompt",
            "reasoningEffort",
            "maxTokens",
            "minimumGeneratedTokens",
            "acceptedFinishReasons",
        }
        assert request["prompt"] in prompts
        assert request["reasoningEffort"] in {"none", "medium", "high"}
        assert 0 < request["minimumGeneratedTokens"] <= request["maxTokens"]
        assert request["maxTokens"] <= document["hostProfile"]["maxOutputTokens"]
        assert request["acceptedFinishReasons"] in (["stop"], ["length"])

    sustained = [
        request for request in requests if request["id"].startswith("sustained-")
    ]
    assert {request["reasoningEffort"] for request in sustained} == {
        "medium",
        "high",
    }
    assert all(request["minimumGeneratedTokens"] > 1719 for request in sustained)
    assert all(request["acceptedFinishReasons"] == ["length"] for request in sustained)

    natural = [
        request for request in requests if request["id"].startswith("sarajevo-")
    ]
    assert {request["reasoningEffort"] for request in natural} == {
        "medium",
        "high",
    }
    assert all(request["acceptedFinishReasons"] == ["stop"] for request in natural)

    print("Qwen 24 GiB release fixture: PASS")


if __name__ == "__main__":
    main()
