# Prompting + function-calling design (interview notes)

## Why this layout

| Piece | Responsibility |
|---|---|
| `providers.py` + `agent_config.json` | Switch DeepSeek / MiniMax / Grok without rewriting the agent. |
| `prompt_versions.py` | Six short, defensible prompt designs for the **same** task. |
| `prompting_lab.py` | Runs each version N times and measures rough consistency. |
| `tools.py` | Tool **schemas** (what the model sees) + **implementations** (what you run). |
| `agent_loop.py` | Day 5–7: one-tool loop (`calculate` only). |
| `working_agent.py` | Day 8–10: 2–3+ tools, error handling, full conversation logs. |
| `app.py` / `demo_cli.py` | Day 11–14: Streamlit + interactive CLI demos. |

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

Key points:

- **Schemas are contracts** — clear `name`, `description`, JSON Schema `parameters`.
- **Models don’t execute tools** — they only propose calls; your loop runs them.
- **Feed results back** with `tool_call_id` so the model can finish.
- **Same code, multiple providers** — only `base_url` / `api_key` / `model` change.

## Working agent (Day 8–10)

Tools:

| Tool | Purpose | Failure mode |
|---|---|---|
| `calculate` | Safe arithmetic | Invalid expression → `ok: false` |
| `convert_units` | Length / mass / temperature | Unsupported units → `ok: false` |
| `get_weather` | Live weather (Open-Meteo) | Unknown city / network → `ok: false` |
| `web_search` | Instant-answer lookup (DDG) | Empty/network → `ok: false` |

Requirements covered:

- **Model switch via config** — `agent_config.json` `"provider"` or `--provider` / `AGENT_PROVIDER`
- **Clear system prompt** — role + when to use tools + what to do on tool errors
- **Graceful tool errors** — tools never raise into the loop; they return JSON `{ok:false,error:...}`; system tells the model not to invent facts
- **Full conversation logs** — `lab_output/conversations/conv_*.json` includes `messages` and a structured `trace` (assistant tool_calls + tool results)

```bash
# edit tool_using_agent/agent_config.json → "provider": "deepseek" | "minimax" | "grok"
python working_agent.py
python working_agent.py --provider deepseek "Convert 100 F to C and weather in London"
```

## Switching providers

```bash
python working_agent.py --provider deepseek "..."
python working_agent.py --provider minimax  "..."
python working_agent.py --provider grok     "..."   # needs XAI_API_KEY
```

If MiniMax returns 401, set `MINIMAX_REGION=cn` when the key is from `platform.minimaxi.com`.
