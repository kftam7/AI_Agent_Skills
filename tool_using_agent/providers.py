"""
Shared OpenAI-compatible clients for MiniMax and DeepSeek.

Switching models = change provider key (or env). Same chat.completions surface.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

MINIMAX_BASE_URLS = {
    "global": "https://api.minimax.io/v1",
    "cn": "https://api.minimaxi.com/v1",
}

PROVIDERS = {
    "minimax": {
        "label": "MiniMax",
        "api_key_env": "MINIMAX_API_KEY",
        "model_env": "MINIMAX_MODEL",
        "default_model": "MiniMax-M3.1-Flash",
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


def clean_secret(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value


def minimax_base_url() -> str:
    explicit = clean_secret(os.getenv("MINIMAX_BASE_URL", ""))
    if explicit:
        return explicit.rstrip("/")
    region = clean_secret(os.getenv("MINIMAX_REGION", "global")).lower()
    if region in ("cn", "china", "mainland", "com"):
        return MINIMAX_BASE_URLS["cn"]
    return MINIMAX_BASE_URLS["global"]


def get_client(provider: str) -> tuple[OpenAI, str, dict]:
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose: {', '.join(PROVIDERS)}")
    cfg = dict(PROVIDERS[provider])
    api_key = clean_secret(os.getenv(cfg["api_key_env"], ""))
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            f"Missing {cfg['api_key_env']}. Copy .env.example → .env and add your key."
        )
    model = clean_secret(os.getenv(cfg["model_env"], cfg["default_model"]))
    base_url = minimax_base_url() if provider == "minimax" else cfg["base_url"]
    cfg["base_url"] = base_url
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    return client, model, cfg


def chat_kwargs(provider: str, cfg: dict) -> dict:
    """Extra create() kwargs that differ by provider (e.g. MiniMax reasoning_effort)."""
    extra: dict = {}
    effort_env = cfg.get("reasoning_effort_env")
    if effort_env:
        effort = clean_secret(
            os.getenv(effort_env, cfg.get("default_reasoning_effort", ""))
        )
        if effort:
            extra["reasoning_effort"] = effort
    return extra
