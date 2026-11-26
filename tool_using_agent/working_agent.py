#!/usr/bin/env python3
"""
Day 8–10: Working multi-tool agent.

Tools: calculate, convert_units, get_weather, web_search
Providers: deepseek | minimax | grok  (switch via agent_config.json or --provider)
Logs the full conversation (including tool calls) under lab_output/conversations/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from providers import chat_kwargs, get_client, load_agent_config
from tools import TOOLS, run_tool

_HERE = Path(__file__).resolve().parent

SYSTEM = (
    "Role: helpful tool-using assistant.\n"
    "Rules:\n"
    "1) Prefer tools for facts you cannot know reliably: weather, unit conversion, "
    "arithmetic, and web lookup.\n"
    "2) If a tool returns ok=false, explain the error briefly and do NOT invent a substitute fact.\n"
    "3) You may call multiple tools when needed, then give a short final answer.\n"
    "4) Cite tool-backed numbers plainly (e.g. temperature in °C, converted values).\n"
    "Tools: calculate, convert_units, get_weather, web_search."
)

DEFAULT_QUESTION = (
    "What's the weather in Tokyo right now in Fahrenheit, "
    "and what is 12 km in miles?"
)


def message_to_dict(message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
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


def _ensure_log_dir(log_dir: str | Path) -> Path:
    path = Path(log_dir)
    if not path.is_absolute():
        path = _HERE / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_conversation_log(log_dir: Path, payload: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    provider = payload.get("provider", "agent")
    path = log_dir / f"conv_{provider}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def run_working_agent(
    provider: str,
    user_question: str,
    *,
    max_steps: int = 8,
    temperature: float = 0.2,
    log_dir: str | Path = "lab_output/conversations",
    verbose: bool = True,
) -> tuple[str, Path]:
    client, model, cfg = get_client(provider)
    extra = chat_kwargs(provider, cfg)
    out_dir = _ensure_log_dir(log_dir)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_question},
    ]
    trace: list[dict[str, Any]] = []

    if verbose:
        print(f"Provider: {cfg['label']}  model={model}  base={cfg['base_url']}")
        print(f"Tools: {', '.join(t['function']['name'] for t in TOOLS)}")
        print(f"User: {user_question}\n")

    final_text = ""
    stopped_reason = "final_answer"

    for step in range(1, max_steps + 1):
        if verbose:
            print("=" * 72)
            print(f"Step {step}: calling model…")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            err = {
                "step": step,
                "event": "model_error",
                "error": f"{type(exc).__name__}: {exc}",
                "body": getattr(exc, "body", None),
            }
            trace.append(err)
            final_text = (
                "I hit a model API error and could not finish. "
                f"Details: {type(exc).__name__}: {exc}"
            )
            stopped_reason = "model_error"
            if verbose:
                print(f"MODEL ERROR: {err['error']}")
            break

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason
        assistant_dict = message_to_dict(msg)
        messages.append(assistant_dict)

        tool_calls = getattr(msg, "tool_calls", None) or []
        step_record: dict[str, Any] = {
            "step": step,
            "event": "assistant",
            "finish_reason": finish,
            "content": msg.content,
            "tool_calls": assistant_dict.get("tool_calls"),
        }
        trace.append(step_record)

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
            stopped_reason = "final_answer"
            break

        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments or "{}"
            result_json = run_tool(name, args)
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            }
            messages.append(tool_msg)
            tool_record = {
                "step": step,
                "event": "tool_result",
                "tool_call_id": tc.id,
                "name": name,
                "arguments": args,
                "result": json.loads(result_json),
            }
            trace.append(tool_record)
            if verbose:
                # highlight failures without crashing the agent
                parsed = tool_record["result"]
                flag = "OK" if parsed.get("ok") else "ERR"
                print(f"tool [{flag}] {name}: {result_json}")
    else:
        final_text = (
            "I reached the step limit before producing a final answer. "
            "Try a simpler question or raise max_steps in agent_config.json."
        )
        stopped_reason = "max_steps"

    log_payload = {
        "provider": provider,
        "model": model,
        "base_url": cfg["base_url"],
        "temperature": temperature,
        "max_steps": max_steps,
        "stopped_reason": stopped_reason,
        "user_question": user_question,
        "final_answer": final_text,
        "system": SYSTEM,
        "tools": [t["function"]["name"] for t in TOOLS],
        "messages": messages,
        "trace": trace,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path = save_conversation_log(out_dir, log_payload)

    if verbose:
        print("=" * 72)
        print("Final answer:")
        print(final_text or "(empty)")
        print()
        print(f"Conversation log: {log_path}")
        print("Interview note: tools can fail; the agent must surface ok=false, not invent.")

    return final_text, log_path


def main() -> int:
    file_cfg = load_agent_config()
    parser = argparse.ArgumentParser(
        description="Day 8–10 multi-tool agent (weather, units, calculate, search)."
    )
    parser.add_argument(
        "--provider",
        default=file_cfg.get("provider", "deepseek"),
        choices=["deepseek", "minimax", "grok"],
        help="Override agent_config.json provider",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="User question (may require multiple tools)",
    )
    parser.add_argument("--max-steps", type=int, default=int(file_cfg.get("max_steps", 8)))
    parser.add_argument(
        "--temperature", type=float, default=float(file_cfg.get("temperature", 0.2))
    )
    parser.add_argument(
        "--log-dir",
        default=str(file_cfg.get("log_dir", "lab_output/conversations")),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        answer, log_path = run_working_agent(
            args.provider,
            args.question,
            max_steps=args.max_steps,
            temperature=args.temperature,
            log_dir=args.log_dir,
            verbose=not args.quiet,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        print(answer)
        print(f"# log: {log_path}", file=sys.stderr)
    return 0 if answer else 1


if __name__ == "__main__":
    sys.exit(main())
