# AI_Agent_Skills

Hands-on exercises for learning AI agent skills.

## Exercise 1 — Compare MiniMax vs DeepSeek

Tiny script under `tool_using_agent/` that calls **MiniMax** and **DeepSeek** with the same prompt and prints a side-by-side comparison. Both APIs are OpenAI Chat Completions–compatible, so switching providers is just `base_url` + `api_key` + `model`.

### Setup

```bash
cd tool_using_agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy env template and add your keys:

```bash
cp ../.env.example ../.env
# edit ../.env → set MINIMAX_API_KEY and DEEPSEEK_API_KEY
```

| Variable | Where to get it |
|---|---|
| `MINIMAX_API_KEY` | [platform.minimax.io](https://platform.minimax.io/user-center/basic-information/interface-key) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |

Optional overrides in `.env`:

```bash
MINIMAX_MODEL=MiniMax-M3.1-Flash   # or MiniMax-M3
MINIMAX_REASONING_EFFORT=low       # low | medium | high | xhigh | max
DEEPSEEK_MODEL=deepseek-chat
```

Do not wrap API keys in quotes.

### Run

```bash
# default prompt
python compare_models.py

# your own prompt
python compare_models.py "What is function calling in one sentence?"

# one provider only
python compare_models.py --only minimax
python compare_models.py --only deepseek
```

### Design notes

- Same prompt, same system message, same temperature/top_p for both models.
- Provider switch lives in the `PROVIDERS` dict in `compare_models.py`.
- Next: a small tool-using agent loop (function calling) reusing the same provider switch.
