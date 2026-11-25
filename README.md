# AI_Agent_Skills

Hands-on exercises for learning AI agent skills.

## Exercise 1 — Compare Grok vs DeepSeek

Tiny script under `tool_using_agent/` that calls **Grok (xAI)** and **DeepSeek** with the same prompt and prints a side-by-side comparison. Both APIs are OpenAI-compatible, so switching providers is just `base_url` + `api_key` + `model`.

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
# edit ../.env → set XAI_API_KEY and DEEPSEEK_API_KEY
```

| Variable | Where to get it |
|---|---|
| `XAI_API_KEY` | [console.x.ai](https://console.x.ai) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |

Optional overrides in `.env`:

```bash
XAI_MODEL=grok-4.7          # or grok-4.6 / grok-4.3 if your key lacks 4.7
XAI_REASONING_EFFORT=low    # low | medium | high | xhigh
DEEPSEEK_MODEL=deepseek-chat
```

Do not wrap API keys in quotes. A Grok `400` usually means a bad key format or an unsupported request field — the script now prints the full error body.

### Run

```bash
# default prompt
python compare_models.py

# your own prompt
python compare_models.py "What is function calling in one sentence?"

# one provider only
python compare_models.py --only grok
python compare_models.py --only deepseek
```

### Design notes

- Same prompt, same system message, same temperature for both models.
- Provider switch lives in the `PROVIDERS` dict in `compare_models.py`.
- Next: a small tool-using agent loop (function calling) reusing the same provider switch.
