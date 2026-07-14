#!/usr/bin/env python3
"""Collect pinned Qwen tokenizer and chat-template golden vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

TEXT_CASES = {
    "ascii": "Hello, world!",
    "italian": "Caffè già, perché l'AI è utile.",
    "cjk": "中文测试：こんにちは。",
    "whitespace": "  one\t two\n\nthree  \n",
    "digits_and_contractions": "I'm 1234, we're coding.",
    "source_code": 'def f(x: int) -> str:\n    return f"{x=}"\n',
    "emoji_zwj_and_nfc": "👩‍💻 café e\u0301",
    "fim_specials": "<|fim_prefix|>left<|fim_suffix|>right<|fim_middle|>",
    "thinking_specials": "<think>penso</think>",
    "tool_specials": "<tool_call>get_weather</tool_call>",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

CHAT_CASES: dict[str, dict[str, Any]] = {
    "plain_thinking": {
        "messages": [{"role": "user", "content": "Quanto fa 17 * 23?"}],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": True},
    },
    "plain_no_thinking": {
        "messages": [{"role": "user", "content": "Rispondi solo: sì"}],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": False},
    },
    "system_and_user": {
        "messages": [
            {"role": "system", "content": "Sei un assistente conciso."},
            {"role": "user", "content": "Saluta in italiano."},
        ],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": True},
    },
    "tools_prompt": {
        "messages": [
            {"role": "system", "content": "Usa gli strumenti quando servono."},
            {"role": "user", "content": "Che tempo fa a Roma?"},
        ],
        "kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tools": TOOLS,
        },
    },
    "tool_roundtrip": {
        "messages": [
            {"role": "user", "content": "Che tempo fa a Roma?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Roma"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature_c":28,"condition":"sunny"}'},
        ],
        "kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": True,
            "tools": TOOLS,
        },
    },
}

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


def collect() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
    )
    if type(tokenizer).__name__ != "Qwen2Tokenizer":
        raise RuntimeError(f"unexpected tokenizer class: {type(tokenizer).__name__}")

    text_vectors = []
    for name, text in TEXT_CASES.items():
        text_vectors.append(
            {
                "name": name,
                "text": text,
                "token_ids": tokenizer.encode(text, add_special_tokens=False),
            }
        )

    chat_vectors = []
    for name, case in CHAT_CASES.items():
        rendered = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, **case["kwargs"]
        )
        chat_vectors.append(
            {
                "name": name,
                "messages": case["messages"],
                "kwargs": case["kwargs"],
                "rendered": rendered,
                "token_ids": tokenizer.encode(rendered, add_special_tokens=False),
            }
        )

    return {
        "source": {"model": MODEL, "revision": REVISION},
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "length": len(tokenizer),
            "model_vocab_size": 248320,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "special_token_ids": {
                token: tokenizer.convert_tokens_to_ids(token)
                for token in SPECIAL_TOKENS
            },
        },
        "text_vectors": text_vectors,
        "chat_vectors": chat_vectors,
    }


def serialize(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("qwen36_tokenizer_chat_golden.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the pinned official tokenizer no longer reproduces the fixture",
    )
    args = parser.parse_args()

    rendered = serialize(collect())
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Qwen reference fixture is stale: {args.output}")
        print(f"Qwen reference fixture matches {MODEL}@{REVISION}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
