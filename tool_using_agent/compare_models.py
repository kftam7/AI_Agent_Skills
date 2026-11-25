#!/usr/bin/env python3
"""
Call Grok (xAI) and DeepSeek with the same prompt and print a side-by-side comparison.

Both APIs are OpenAI-compatible — switching providers is just base_url + api_key + model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from repo root (parent of this folder) or cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()  # also allow cwd override

DEFAULT_PROMPT = (
    "In 3 short bullets, explain what a tool-using LLM agent is "
    "and why function calling matters."
)

# Provider configs — change MODEL env vars or edit these to switch models
PROVIDERS = {
    "grok": {
        "label": "Grok (xAI)",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model_env": "XAI_MODEL",
        "default_model": "grok-4.7",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
}


def make_client(provider_key: str) -> tuple[OpenAI, str, str]:
    cfg = PROVIDERS[provider_key]
    api_key = os.getenv(cfg["api_key_env"], "").strip()
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            f"Missing {cfg['api_key_env']}. "
            f"Copy .env.example → .env and add your key."
        )
    model = os.getenv(cfg["model_env"], cfg["default_model"]).strip()
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return client, model, cfg["label"]


def ask(provider_key: str, prompt: str) -> dict:
    client, model, label = make_client(provider_key)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise technical tutor. Be clear and direct.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        return {
            "ok": True,
            "label": label,
            "model": model,
            "text": text,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — show any API/network error clearly
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "label": label,
            "model": model,
            "text": "",
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def print_result(result: dict) -> None:
    print("=" * 72)
    print(f"{result['label']}  ·  model={result['model']}  ·  {result['elapsed_ms']} ms")
    if result["prompt_tokens"] is not None:
        print(
            f"tokens: prompt={result['prompt_tokens']}  "
            f"completion={result['completion_tokens']}"
        )
    print("-" * 72)
    if result["ok"]:
        print(result["text"] or "(empty response)")
    else:
        print(f"ERROR: {result['error']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Grok and DeepSeek answers to the same prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send to both models (default: a short agent explainer)",
    )
    parser.add_argument(
        "--only",
        choices=list(PROVIDERS),
        help="Call only one provider instead of both",
    )
    args = parser.parse_args()

    keys = [args.only] if args.only else list(PROVIDERS)
    print(f"Prompt:\n{args.prompt}\n")

    results = [ask(key, args.prompt) for key in keys]
    for result in results:
        print_result(result)

    ok = [r for r in results if r["ok"]]
    if len(ok) == 2:
        print("=" * 72)
        print("Comparison notes")
        print("-" * 72)
        a, b = ok
        print(f"• Latency: {a['label']} {a['elapsed_ms']} ms vs {b['label']} {b['elapsed_ms']} ms")
        print(f"• Length:  {a['label']} {len(a['text'])} chars vs {b['label']} {len(b['text'])} chars")
        print("• Read both answers above — same prompt, same system message, temperature=0.3.")
        print()

    failed = [r for r in results if not r["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
