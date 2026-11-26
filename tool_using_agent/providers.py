"""
Shared OpenAI-compatible clients.

Supported providers: deepseek, minimax, grok (xAI).
Switch via agent_config.json "provider" or --provider / AGENT_PROVIDER env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

CONFIG_PATH = _HERE / "agent_config.json"

MINIMAX_BASE_URLS = {
    "global": "https://api.minimax.io/v1",
    "cn": "https://api.minimaxi.com/v1",
}

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "minimax": {
        "label": "MiniMax",
        "api_key_env": "MINIMAX_API_KEY",
        "model_env": "MINIMAX_MODEL",
        "default_model": "MiniMax-M3.1-Flash",
        "reasoning_effort_env": "MINIMAX_REASONING_EFFORT",
        "default_reasoning_effort": "low",
    },
    "grok": {
        "label": "Grok (xAI)",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model_env": "XAI_MODEL",
        "default_model": "grok-4.7",
        # Prefer chat completions for tool parity with DeepSeek; Responses also works
        # but tool-loop examples are Chat Completions–shaped in this repo.
    },
}


def clean_secret(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value


def load_agent_config() -> dict:
    cfg: dict = {
        "provider": "deepseek",
        "max_steps": 8,
        "temperature": 0.2,
        "log_dir": "lab_output/conversations",
    }
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if not str(k).startswith("comment")})
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {CONFIG_PATH}: {exc}") from exc
    env_provider = clean_secret(os.getenv("AGENT_PROVIDER", ""))
    if env_provider:
        cfg["provider"] = env_provider
    return cfg


def minimax_base_url() -> str:
    explicit = clean_secret(os.getenv("MINIMAX_BASE_URL", ""))
    if explicit:
        return explicit.rstrip("/")
    region = clean_secret(os.getenv("MINIMAX_REGION", "global")).lower()
    if region in ("cn", "china", "mainland", "com"):
        return MINIMAX_BASE_URLS["cn"]
    return MINIMAX_BASE_URLS["global"]


def resolve_base_url(provider: str, cfg: dict) -> str:
    if provider == "minimax":
        return minimax_base_url()
    return cfg["base_url"]


def get_client(provider: str) -> tuple[OpenAI, str, dict]:
    if provider not in PROVIDERS:
        raise SystemExit(
            f"Unknown provider {provider!r}. Choose: {', '.join(PROVIDERS)}"
        )
    cfg = dict(PROVIDERS[provider])
    api_key = clean_secret(os.getenv(cfg["api_key_env"], ""))
    if not api_key or api_key.startswith("your_"):
        raise SystemExit(
            f"Missing {cfg['api_key_env']}. Copy .env.example → .env and add your key."
        )
    model = clean_secret(os.getenv(cfg["model_env"], cfg["default_model"]))
    base_url = resolve_base_url(provider, cfg)
    cfg["base_url"] = base_url
    timeout = 180.0 if provider == "grok" else 120.0
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return client, model, cfg


def chat_kwargs(provider: str, cfg: dict) -> dict:
    """Extra create() kwargs that differ by provider."""
    extra: dict = {}
    effort_env = cfg.get("reasoning_effort_env")
    if effort_env:
        effort = clean_secret(
            os.getenv(effort_env, cfg.get("default_reasoning_effort", ""))
        )
        if effort:
            extra["reasoning_effort"] = effort
    return extra
