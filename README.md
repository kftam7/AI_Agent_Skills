# AI_Agent_Skills

A small, interview-ready **tool-using agent** lab: short defensible prompts, an explicit function-calling loop, and the same code path across **DeepSeek**, **MiniMax**, and optional **Grok (xAI)**.

This repo is the public proof for a hands-on “AI agent skills” path (compare models → prompt lab → one-tool loop → multi-tool agent → demo UI).

---

## Quick demo

```bash
cd tool_using_agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env         # add DEEPSEEK_API_KEY (and others you use)
```

**Streamlit UI**

```bash
streamlit run app.py --server.port 8505
```

**Interactive CLI**

```bash
python demo_cli.py --provider deepseek
```

**One-shot agent**

```bash
python working_agent.py "What's the weather in Tokyo in F, and how many miles is 12 km?"
```

Switch models in `agent_config.json` (`"provider": "deepseek" | "minimax" | "grok"`) or pass `--provider`.

| Variable | Source |
|---|---|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `MINIMAX_API_KEY` | [platform.minimax.io](https://platform.minimax.io) or China [platform.minimaxi.com](https://platform.minimaxi.com) (`MINIMAX_REGION=cn`) |
| `XAI_API_KEY` (optional) | [console.x.ai](https://console.x.ai) |

---

## Why this prompt structure

I keep prompts **short** and **structured** so every line is defensible in an interview:

1. **System = policy, user = task**  
   Role, rules, and output expectations live in the system message. Changing questions does not rewrite policy.

2. **Role + rules + format** (not a long essay)  
   Example pattern used in the labs and agent:
   - Role: who the model is  
   - Rules: grounding, when to call tools, what to do on failure  
   - Format / behavior: short final answers; don’t invent tool facts  

3. **Explicit failure modes**  
   For missing product facts (prompting lab) and for `ok: false` tool results (working agent), the prompt says **admit unknown / report the error** — not “be helpful” in a way that invites hallucination.

4. **Evidence over vibes**  
   `prompting_lab.py` runs **six versions of the same task** and measures rough consistency so you can say “I kept v5/v6 because constraints + unknown-handling were more stable,” not “it felt better.”

See `tool_using_agent/prompt_versions.py` and `DESIGN.md`.

---

## How function calling is implemented

The model never executes tools. **Your loop does.**

```
messages + tool schemas
        ↓
   chat.completions (tool_choice=auto)
        ↓
   tool_calls? ──yes──► run_tool() → append role=tool → loop again
        │
        no
        ↓
   final natural-language answer (+ save JSON log)
```

| Piece | Job |
|---|---|
| `tools.py` | JSON Schema **contracts** + local implementations (`calculate`, `convert_units`, `get_weather`, `web_search`) |
| `agent_loop.py` | Day 5–7 teaching loop with **one** tool |
| `working_agent.py` | Day 8–10 multi-tool loop, graceful tool errors, full conversation log |
| `providers.py` | OpenAI-compatible clients; switch provider without rewriting the loop |

Tool results are always JSON. Failures look like `{ "ok": false, "error": "..." }` so the agent can surface them instead of crashing or inventing numbers.

Logs (including tool calls) are written to `tool_using_agent/lab_output/conversations/conv_*.json`.

---

## Differences observed: Grok vs DeepSeek (and MiniMax)

This lab targets **provider-portable** agents. Practical notes from building and running it:

| Topic | Observation |
|---|---|
| **API shape** | DeepSeek and MiniMax fit the OpenAI **Chat Completions + tools** loop cleanly. Grok/xAI is also OpenAI-compatible on Chat Completions, but newer Grok docs push the **Responses** API for reasoning features; keep the tool loop on Chat Completions for parity unless you deliberately migrate. |
| **Auth / access** | DeepSeek was the most straightforward “key in `.env` → tools work” path in this project. Grok failed here with **incorrect API key** until a valid `XAI_API_KEY` from [console.x.ai](https://console.x.ai) is available. MiniMax often fails with **401 invalid api key (2049)** when the key region does not match the host (`api.minimax.io` vs `api.minimaxi.com`) — fix with `MINIMAX_REGION`. |
| **Reasoning extras** | MiniMax accepts `reasoning_effort`; Grok reasoning models care about effort/sampling constraints. DeepSeek tool demos are simplest with thinking disabled or a non-thinking chat model when you want predictable tool_calls. |
| **What to claim publicly** | Prefer: *“same agent loop, config-switched providers.”* Only claim quality rankings (latency, tool reliability, answer style) after you run the same questions on keys you control and keep the JSON logs. |

If you only have DeepSeek today, that still satisfies the architecture goal: **one loop, many providers**. Add Grok when the key works; no redesign required.

---

## Exercise map

| Days | Script / UI | Focus |
|---|---|---|
| 1–2 | `compare_models.py` | Same prompt, two providers |
| 3–4 | `prompting_lab.py` | Short, defensible prompts |
| 5–7 | `agent_loop.py` | One-tool function-calling loop |
| 8–10 | `working_agent.py` | 2–3+ tools, errors, conversation logs |
| 11–14 | `app.py` / `demo_cli.py` | Demo UI + this README |

---

## Layout

```
tool_using_agent/
  agent_config.json   # default provider / temperature / max_steps / log_dir
  providers.py
  tools.py
  agent_loop.py       # 1-tool teaching loop
  working_agent.py    # multi-tool agent + logs
  app.py              # Streamlit demo
  demo_cli.py         # interactive CLI
  prompting_lab.py / prompt_versions.py
  compare_models.py
  DESIGN.md
  lab_output/         # gitignored run artifacts
```

---

## License / intent

Educational portfolio project for learning tool-using agents. Bring your own API keys; never commit `.env`.
