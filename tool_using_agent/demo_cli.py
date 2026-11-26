#!/usr/bin/env python3
"""
Day 11–14: Interactive CLI demo for the working agent.

  python demo_cli.py
  python demo_cli.py --provider deepseek
"""

from __future__ import annotations

import argparse
import sys

from providers import load_agent_config
from working_agent import run_working_agent

BANNER = """
AI Agent Skills — interactive tool agent
Commands:  /provider deepseek|minimax|grok   /quit
Tools:     calculate, convert_units, get_weather, web_search
""".strip()


def main() -> int:
    file_cfg = load_agent_config()
    parser = argparse.ArgumentParser(description="Interactive CLI for the working agent.")
    parser.add_argument(
        "--provider",
        default=file_cfg.get("provider", "deepseek"),
        choices=["deepseek", "minimax", "grok"],
    )
    parser.add_argument("--max-steps", type=int, default=int(file_cfg.get("max_steps", 8)))
    parser.add_argument(
        "--temperature", type=float, default=float(file_cfg.get("temperature", 0.2))
    )
    args = parser.parse_args()
    provider = args.provider

    print(BANNER)
    print(f"Provider: {provider}  (change with /provider …)\n")

    while True:
        try:
            line = input(f"[{provider}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not line:
            continue
        if line in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.")
            return 0
        if line.startswith("/provider"):
            parts = line.split()
            if len(parts) != 2 or parts[1] not in {"deepseek", "minimax", "grok"}:
                print("Usage: /provider deepseek|minimax|grok")
                continue
            provider = parts[1]
            print(f"Switched provider → {provider}")
            continue

        try:
            answer, log_path = run_working_agent(
                provider,
                line,
                max_steps=args.max_steps,
                temperature=args.temperature,
                verbose=True,
            )
            if not answer:
                print("(empty final answer)")
            print(f"(log: {log_path})\n")
        except SystemExit as exc:
            print(f"Config/key error: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
