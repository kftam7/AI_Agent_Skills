# AI_Agent_Skills

Hands-on exercises for learning AI agent skills.

Providers: **DeepSeek**, **MiniMax**, and optional **Grok (xAI)** — switch via `agent_config.json` or `--provider`.

## Setup

```bash
cd tool_using_agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env  # then add keys
```

| Variable | Where to get it |
|---|---|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `MINIMAX_API_KEY` | Global: [platform.minimax.io](https://platform.minimax.io) · China: [platform.minimaxi.com](https://platform.minimaxi.com) |
| `XAI_API_KEY` (optional) | [console.x.ai](https://console.x.ai) |

MiniMax region (key and host must match): `MINIMAX_REGION=global` or `cn`.

Default provider: edit `tool_using_agent/agent_config.json`.

---

## Exercise 1 — Compare models

```bash
python compare_models.py
python compare_models.py --only deepseek
```

---

## Exercise 2 — Day 3–4: Short, defensible prompting

```bash
python prompting_lab.py --provider deepseek --runs 3
```

See `prompt_versions.py` and fill notes under `lab_output/`.

---

## Exercise 3 — Day 5–7: Function calling basics (1 tool)

```bash
python agent_loop.py --provider deepseek
python agent_loop.py --provider minimax "What is (81 / 9) ** 2 + 3?"
```

---

## Exercise 4 — Day 8–10: Working multi-tool agent

Tools: **weather**, **unit conversion**, **calculator**, optional **web search**.

```bash
# uses provider from agent_config.json
python working_agent.py

# override provider
python working_agent.py --provider deepseek
python working_agent.py --provider minimax
python working_agent.py --provider grok   # if you have XAI_API_KEY

# multi-tool question
python working_agent.py "What's the weather in Tokyo in Fahrenheit, and how many miles is 12 km?"
```

Each run writes a full conversation log (messages + tool calls + tool results) to:

`lab_output/conversations/conv_<provider>_<timestamp>.json`

Tool errors return `{ "ok": false, "error": "..." }` — the agent is instructed to report them, not invent facts.

Offline tool smoke test (no LLM key):

```bash
python -c "
from tools import calculate, convert_units, run_tool
print(calculate('(17.5*4)+9/3'))
print(convert_units(12, 'km', 'mi'))
print(run_tool('convert_units', '{\"value\":100,\"from_unit\":\"C\",\"to_unit\":\"F\"}'))
"
```

---

## Layout

```
tool_using_agent/
  agent_config.json  # provider + max_steps + temperature + log_dir
  providers.py       # DeepSeek / MiniMax / Grok clients
  compare_models.py
  prompt_versions.py / prompting_lab.py
  tools.py           # calculate, convert_units, get_weather, web_search
  agent_loop.py      # Day 5–7 one-tool loop
  working_agent.py   # Day 8–10 multi-tool + logs
  DESIGN.md
  lab_output/
```
