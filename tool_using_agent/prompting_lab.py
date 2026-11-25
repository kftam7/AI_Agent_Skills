#!/usr/bin/env python3
"""
Day 3–4: Short, defensible prompting lab.

Runs 6 prompt versions of the SAME task, each N times, so you can see which
designs are more consistent. Writes a results JSON + a notes template.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prompt_versions import QUESTION, VERSIONS
from providers import chat_kwargs, get_client

OUT_DIR = Path(__file__).resolve().parent / "lab_output"


def run_once(client, model: str, extra: dict, system: str | None, user: str, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def consistency_stats(texts: list[str]) -> dict:
    norms = [normalize(t) for t in texts]
    counts = Counter(norms)
    most_common, freq = counts.most_common(1)[0]
    return {
        "unique_normalized": len(counts),
        "most_common_count": freq,
        "consistency_ratio": round(freq / len(texts), 3) if texts else 0.0,
        "most_common_preview": most_common[:200],
    }


def score_unknown_handling(text: str) -> dict:
    """Lightweight checks for v5/v6-style honesty about dishwasher safety."""
    lower = text.lower()
    invented_yes = bool(re.search(r"dishwasher[^\n.]{0,40}\byes\b", lower)) or (
        "dishwasher-safe" in lower and "yes" in lower and "unknown" not in lower
    )
    says_unknown = "i don't know" in lower or "unknown" in lower or "not listed" in lower
    mentions_colors = "black" in lower and "blue" in lower
    return {
        "mentions_stock_colors": mentions_colors,
        "admits_unknown_dishwasher": says_unknown and not invented_yes,
        "likely_invented_dishwasher_yes": invented_yes and not says_unknown,
    }


def write_notes_template(path: Path, result_path: Path) -> None:
    path.write_text(
        f"""# Prompting lab notes (fill this in)

Results file: `{result_path.name}`
Task question: {QUESTION}

## Interview soundbite

> I keep prompts short and structured: **role + rules + output format** in the
> system message, task data in the user message. I add an explicit **I don't know**
> rule for missing facts so the model fails safely instead of inventing.

## Version ranking (your judgment)

| Version | Consistency (from script) | Quality / honesty | Keep? | Why |
|---|---|---|---|---|
| v1_vague_user_only | | | | |
| v2_role_in_user | | | | |
| v3_system_role_loose_user | | | | |
| v4_structured_system | | | | |
| v5_explicit_constraints | | | | |
| v6_unknown_handling | | | | |

## What I learned

- System vs user: …
- Explicit output constraints: …
- Handling “I don’t know”: …

## Defensible final prompt (paste the winner)

```
SYSTEM:
…

USER:
…
```
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day 3–4 prompting lab.")
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "minimax"],
        help="Which model to use for the lab (default: deepseek)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Repeats per version (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional version ids to run (default: all)",
    )
    args = parser.parse_args()

    client, model, cfg = get_client(args.provider)
    extra = chat_kwargs(args.provider, cfg)
    versions = VERSIONS
    if args.only:
        wanted = set(args.only)
        versions = [v for v in VERSIONS if v["id"] in wanted]
        if not versions:
            raise SystemExit(f"No matching versions for {args.only}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = OUT_DIR / f"prompting_{args.provider}_{stamp}.json"
    notes_path = OUT_DIR / f"prompting_notes_{args.provider}_{stamp}.md"

    print(f"Provider: {cfg['label']}  model={model}  runs/version={args.runs}")
    print(f"Question: {QUESTION}\n")

    report = {
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "runs": args.runs,
        "question": QUESTION,
        "versions": [],
    }

    for ver in versions:
        print("=" * 72)
        print(f"{ver['id']}  ·  {ver['focus']}")
        print(f"defend: {ver['defend']}")
        print("-" * 72)
        texts: list[str] = []
        for i in range(args.runs):
            started = time.perf_counter()
            try:
                text = run_once(
                    client, model, extra, ver["system"], ver["user"], args.temperature
                )
                ms = int((time.perf_counter() - started) * 1000)
                texts.append(text)
                print(f"[run {i + 1}/{args.runs} · {ms} ms]\n{text}\n")
            except Exception as exc:  # noqa: BLE001
                err = f"ERROR: {type(exc).__name__}: {exc}"
                texts.append(err)
                print(f"[run {i + 1}/{args.runs}]\n{err}\n")

        stats = consistency_stats(texts)
        honesty = [score_unknown_handling(t) for t in texts if not t.startswith("ERROR:")]
        entry = {
            "id": ver["id"],
            "focus": ver["focus"],
            "defend": ver["defend"],
            "system": ver["system"],
            "user": ver["user"],
            "outputs": texts,
            "consistency": stats,
            "honesty_checks": honesty,
        }
        report["versions"].append(entry)
        print(
            f"consistency: {stats['consistency_ratio']} "
            f"({stats['most_common_count']}/{args.runs} identical after normalize; "
            f"{stats['unique_normalized']} unique)\n"
        )

    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_notes_template(notes_path, result_path)
    print(f"Wrote {result_path}")
    print(f"Fill in {notes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
