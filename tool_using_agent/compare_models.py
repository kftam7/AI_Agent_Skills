#!/usr/bin/env python3
"""
Call MiniMax and DeepSeek with the same prompt and print a side-by-side comparison.

Uses shared providers.py — switching is still just base_url + api_key + model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from providers import PROVIDERS, chat_kwargs, get_client

DEFAULT_PROMPT = (
    "In 3 short bullets, explain what a tool-using LLM agent is "
    "and why function calling matters."
)

TEMPERATURE = 0.7
TOP_P = 0.95
SYSTEM = "You are a concise technical tutor. Be clear and direct."


def format_api_error(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append("body: " + json.dumps(body, indent=2, default=str))
        except TypeError:
            parts.append(f"body: {body!r}")
    return "\n".join(parts)


def ask(provider_key: str, prompt: str) -> dict:
    client, model, cfg = get_client(provider_key)
    label = cfg["label"]
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            **chat_kwargs(provider_key, cfg),
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "label": label,
            "model": model,
            "base_url": cfg["base_url"],
            "text": text,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "label": label,
            "model": model,
            "base_url": cfg["base_url"],
            "text": "",
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": format_api_error(exc),
        }


def print_result(result: dict) -> None:
    print("=" * 72)
    print(
        f"{result['label']}  ·  model={result['model']}  ·  "
        f"{result['elapsed_ms']} ms  ·  base={result['base_url']}"
    )
    if result["prompt_tokens"] is not None:
        print(
            f"tokens: prompt={result['prompt_tokens']}  "
            f"completion={result['completion_tokens']}"
        )
    print("-" * 72)
    if result["ok"]:
        print(result["text"] or "(empty response)")
    else:
        print(f"ERROR:\n{result['error']}")
        if result["label"] == "MiniMax" and "401" in (result["error"] or ""):
            print(
                "\nMiniMax 401 hint: key and host must be from the SAME region.\n"
                "  • Key from platform.minimax.io  → MINIMAX_REGION=global\n"
                "  • Key from platform.minimaxi.com → MINIMAX_REGION=cn"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax and DeepSeek answers to the same prompt."
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--only", choices=list(PROVIDERS))
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
        print(
            f"• Same prompt + system message; temperature={TEMPERATURE}, top_p={TOP_P}."
        )
        print()

    return 1 if any(not r["ok"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
