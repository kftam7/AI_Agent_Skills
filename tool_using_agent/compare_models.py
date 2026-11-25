#!/usr/bin/env python3
"""
Call MiniMax and DeepSeek with the same prompt and print a side-by-side comparison.

Both APIs are OpenAI Chat Completions–compatible — switching providers is
just base_url + api_key + model.
"""

from __future__ import annotations

import argparse
import json
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

TEMPERATURE = 0.7
TOP_P = 0.95
SYSTEM = "You are a concise technical tutor. Be clear and direct."

# MiniMax runs two separate platforms; key and host must match.
MINIMAX_BASE_URLS = {
    "global": "https://api.minimax.io/v1",  # key from platform.minimax.io
    "cn": "https://api.minimaxi.com/v1",  # key from platform.minimaxi.com
}

# Provider configs — change MODEL env vars or edit these to switch models
PROVIDERS = {
    "minimax": {
        "label": "MiniMax",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_fn": "minimax",  # resolved in make_client
        "model_env": "MINIMAX_MODEL",
        "default_model": "MiniMax-M3.1-Flash",
        # Optional thinking depth: low | medium | high | xhigh | max
        "reasoning_effort_env": "MINIMAX_REASONING_EFFORT",
        "default_reasoning_effort": "low",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
}


def _clean_secret(value: str) -> str:
    """Strip whitespace and optional surrounding quotes from .env values."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value


def minimax_base_url() -> str:
    """Resolve MiniMax host from MINIMAX_BASE_URL or MINIMAX_REGION."""
    explicit = _clean_secret(os.getenv("MINIMAX_BASE_URL", ""))
    if explicit:
        return explicit.rstrip("/")
    region = _clean_secret(os.getenv("MINIMAX_REGION", "global")).lower()
    if region in ("cn", "china", "mainland", "com"):
        return MINIMAX_BASE_URLS["cn"]
    return MINIMAX_BASE_URLS["global"]


def make_client(provider_key: str) -> tuple[OpenAI, str, dict]:
    cfg = dict(PROVIDERS[provider_key])  # shallow copy so we can attach resolved url
    api_key = _clean_secret(os.getenv(cfg["api_key_env"], ""))
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            f"Missing {cfg['api_key_env']}. "
            f"Copy .env.example → .env and add your key."
        )
    model = _clean_secret(os.getenv(cfg["model_env"], cfg["default_model"]))
    if cfg.get("base_url_fn") == "minimax":
        cfg["base_url"] = minimax_base_url()
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=120.0)
    return client, model, cfg


def format_api_error(exc: BaseException) -> str:
    """Prefer the provider JSON body over a truncated one-liner."""
    parts = [f"{type(exc).__name__}: {exc}"]
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append("body: " + json.dumps(body, indent=2, default=str))
        except TypeError:
            parts.append(f"body: {body!r}")
    response = getattr(exc, "response", None)
    if response is not None and body is None:
        try:
            parts.append("body: " + json.dumps(response.json(), indent=2, default=str))
        except Exception:  # noqa: BLE001
            text = getattr(response, "text", None)
            if text:
                parts.append(f"body: {text}")
    return "\n".join(parts)


def ask(provider_key: str, prompt: str) -> dict:
    client, model, cfg = make_client(provider_key)
    label = cfg["label"]
    started = time.perf_counter()
    try:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        # MiniMax-M3.1-Flash accepts reasoning_effort; DeepSeek ignores if omitted
        effort_env = cfg.get("reasoning_effort_env")
        if effort_env:
            effort = _clean_secret(
                os.getenv(effort_env, cfg.get("default_reasoning_effort", ""))
            )
            if effort:
                kwargs["reasoning_effort"] = effort

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = (message.content or "").strip()
        # MiniMax may also return thinking in reasoning_content; keep answer-only
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
    except Exception as exc:  # noqa: BLE001 — show any API/network error clearly
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
    extra = f"  ·  base={result['base_url']}" if result.get("base_url") else ""
    print(
        f"{result['label']}  ·  model={result['model']}  ·  "
        f"{result['elapsed_ms']} ms{extra}"
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
                "  • Key from platform.minimax.io  → MINIMAX_REGION=global "
                "(api.minimax.io)  [default]\n"
                "  • Key from platform.minimaxi.com → MINIMAX_REGION=cn "
                "(api.minimaxi.com)\n"
                "Also check: no quotes in .env, key not revoked, account has credit."
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax and DeepSeek answers to the same prompt."
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
        print(
            f"• Same prompt + system message; temperature={TEMPERATURE}, top_p={TOP_P}."
        )
        print()

    failed = [r for r in results if not r["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
