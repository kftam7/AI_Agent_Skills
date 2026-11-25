# Prompting + function-calling design (interview notes)

## Why this layout

| Piece | Responsibility |
|---|---|
| `providers.py` | One place for MiniMax / DeepSeek clients. Switch models without rewriting the agent. |
| `prompt_versions.py` | Six short, defensible prompt designs for the **same** task. |
| `prompting_lab.py` | Runs each version N times and measures rough consistency. |
| `tools.py` | Tool **schema** (what the model sees) + **implementation** (what you run). |
| `agent_loop.py` | The full tool loop: call → parse → execute → feed back → final answer. |

## Prompting (Day 3–4) — what to say in an interview

**Format I use:** short **system** = role + rules + output format; **user** = task data only.

Why:

1. **System vs user** — Policy stays stable; tasks change. Easier to reuse and A/B test.
2. **Explicit output constraints** — Makes answers checkable (parse lines / JSON), so “consistent” is measurable.
3. **I don’t know** — Missing facts are a failure mode. An explicit rule beats hoping the model refuses.

The lab’s `v6_unknown_handling` is usually the easiest to defend; `v5_explicit_constraints` is the easiest to score automatically.

## Function calling (Day 5–7) — the loop

```
User question
    ↓
Model (+ tools schemas)  →  maybe tool_calls
    ↓
Your code runs the tool(s)
    ↓
Append role=tool results
    ↓
Model again → final natural-language answer
```

Key points to practice / defend:

- **Schemas are contracts** — clear `name`, `description`, JSON Schema `parameters`.
- **Models don’t execute tools** — they only propose calls; your loop runs them.
- **Feed results back** with `tool_call_id` so the model can finish.
- **Same code, two providers** — only `base_url` / `api_key` / `model` change (`--provider`).

## Switching MiniMax ↔ DeepSeek

```bash
python agent_loop.py --provider deepseek "What is 21 * 19?"
python agent_loop.py --provider minimax  "What is 21 * 19?"
```

If MiniMax returns 401, set `MINIMAX_REGION=cn` when the key is from `platform.minimaxi.com`.
