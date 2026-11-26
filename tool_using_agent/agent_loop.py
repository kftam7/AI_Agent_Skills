#!/usr/bin/env python3
"""
Day 5–7: Function-calling agent loop (one tool: calculate).

Flow (same for MiniMax and DeepSeek — only provider/model changes):
  1. Send messages + tool schemas
  2. If model returns tool_calls → parse → execute → append tool results
  3. Repeat until a normal assistant message (final answer) or max steps
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from providers import chat_kwargs, get_client
from tools import CALCULATE_TOOLS, run_tool

SYSTEM = (
    "Role: careful math assistant.\n"
    "Rules:\n"
    "1) For any arithmetic, call the calculate tool — do not invent results.\n"
    "2) After you get tool results, give a short final answer in plain language.\n"
    "3) If the question is not about calculation, say so briefly."
)

DEFAULT_QUESTION = "What is (17.5 * 4) + 9 / 3? Show the exact number."


def message_to_dict(message) -> dict[str, Any]:
    """Convert SDK message object into a plain dict safe to re-send."""
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tool_calls
        ]
    return data


def run_agent(
    provider: str,
    user_question: str,
    *,
    max_steps: int = 5,
    temperature: float = 0.2,
    verbose: bool = True,
) -> str:
    client, model, cfg = get_client(provider)
    extra = chat_kwargs(provider, cfg)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_question},
    ]

    if verbose:
        print(f"Provider: {cfg['label']}  model={model}  base={cfg['base_url']}")
        print(f"User: {user_question}\n")

    final_text = ""
    for step in range(1, max_steps + 1):
        if verbose:
            print("=" * 72)
            print(f"Step {step}: calling model…")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=CALCULATE_TOOLS,
            tool_choice="auto",
            temperature=temperature,
            **extra,
        )
        msg = response.choices[0].message
        finish = response.choices[0].finish_reason
        messages.append(message_to_dict(msg))

        tool_calls = getattr(msg, "tool_calls", None) or []
        if verbose:
            print(f"finish_reason={finish}")
            if msg.content:
                print(f"assistant content:\n{msg.content}")
            if tool_calls:
                print(f"tool_calls ({len(tool_calls)}):")
                for tc in tool_calls:
                    print(f"  - {tc.function.name}({tc.function.arguments})")

        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        # Execute each tool call and feed results back
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments or "{}"
            result_json = run_tool(name, args)
            if verbose:
                print(f"tool result [{name}]: {result_json}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                }
            )
    else:
        final_text = "(stopped: max steps reached without a final answer)"

    if verbose:
        print("=" * 72)
        print("Final answer:")
        print(final_text or "(empty)")
        print()
        print("Interview note: the model never executed code — your loop did.")
    return final_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-tool function-calling agent (calculate)."
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "minimax", "grok"],
        help="Swap providers with almost no code change (default: deepseek)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="User question that should trigger calculate",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true", help="Only print final answer")
    args = parser.parse_args()

    try:
        answer = run_agent(
            args.provider,
            args.question,
            max_steps=args.max_steps,
            temperature=args.temperature,
            verbose=not args.quiet,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        body = getattr(exc, "body", None)
        if body is not None:
            print(json.dumps(body, indent=2, default=str), file=sys.stderr)
        return 1

    if args.quiet:
        print(answer)
    return 0 if answer else 1


if __name__ == "__main__":
    sys.exit(main())
