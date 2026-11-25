#!/usr/bin/env python3
"""
Call Grok (xAI) and DeepSeek with the same prompt and print a side-by-side comparison.

Both APIs are OpenAI-compatible. Switching providers is just base_url + api_key + model.
Grok uses the Responses API (recommended for grok-4.7); DeepSeek uses Chat Completions.
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

# Shared sampling — xAI recommends ~0.7 / 0.95 for Grok comparisons
TEMPERATURE = 0.7
TOP_P = 0.95
SYSTEM = "You are a concise technical tutor. Be clear and direct."

# Provider configs — change MODEL env vars or edit these to switch models
PROVIDERS = {
    "grok": {
        "label": "Grok (xAI)",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model_env": "XAI_MODEL",
        "default_model": "grok-4.7",
        # Responses API is the supported path for current Grok reasoning models
        "api": "responses",
        # Keep comparison snappy; override with XAI_REASONING_EFFORT=high if needed
        "reasoning_effort_env": "XAI_REASONING_EFFORT",
        "default_reasoning_effort": "low",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
        "api": "chat",
    },
}


def _clean_secret(value: str) -> str:
    """Strip whitespace and optional surrounding quotes from .env values."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value


def make_client(provider_key: str) -> tuple[OpenAI, str, dict]:
    cfg = PROVIDERS[provider_key]
    api_key = _clean_secret(os.getenv(cfg["api_key_env"], ""))
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            f"Missing {cfg['api_key_env']}. "
            f"Copy .env.example → .env and add your key."
        )
    model = _clean_secret(os.getenv(cfg["model_env"], cfg["default_model"]))
    # Reasoning models can take a while; give Grok a longer timeout
    timeout = 120.0 if cfg["api"] == "responses" else 60.0
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=timeout)
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


def _usage_tokens(usage) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", None)
    return prompt, completion


def _text_from_responses(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) in ("output_text", "text"):
                chunks.append(getattr(part, "text", "") or "")
    return "".join(chunks).strip()


def ask(provider_key: str, prompt: str) -> dict:
    client, model, cfg = make_client(provider_key)
    label = cfg["label"]
    started = time.perf_counter()
    try:
        if cfg["api"] == "responses":
            effort = _clean_secret(
                os.getenv(
                    cfg.get("reasoning_effort_env", ""),
                    cfg.get("default_reasoning_effort", "low"),
                )
            )
            kwargs = {
                "model": model,
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
            if effort:
                kwargs["reasoning"] = {"effort": effort}
            response = client.responses.create(**kwargs)
            text = _text_from_responses(response)
            prompt_tokens, completion_tokens = _usage_tokens(response.usage)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            text = (response.choices[0].message.content or "").strip()
            prompt_tokens, completion_tokens = _usage_tokens(response.usage)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "label": label,
            "model": model,
            "api": cfg["api"],
            "text": text,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — show any API/network error clearly
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "label": label,
            "model": model,
            "api": cfg["api"],
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
        f"api={result['api']}  ·  {result['elapsed_ms']} ms"
    )
    if result["prompt_tokens"] is not None:
        print(
            f"tokens: prompt/input={result['prompt_tokens']}  "
            f"completion/output={result['completion_tokens']}"
        )
    print("-" * 72)
    if result["ok"]:
        print(result["text"] or "(empty response)")
    else:
        print(f"ERROR:\n{result['error']}")
        if result["label"].startswith("Grok"):
            print(
                "\nHints: check XAI_API_KEY (no quotes), try XAI_MODEL=grok-4.6, "
                "or XAI_REASONING_EFFORT=low in .env"
            )
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
        print(
            f"• Same prompt + system message; temperature={TEMPERATURE}, top_p={TOP_P}."
        )
        print()

    failed = [r for r in results if not r["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
