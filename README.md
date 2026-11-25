# AI_Agent_Skills

Hands-on exercises for learning AI agent skills (MiniMax + DeepSeek).

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
| `MINIMAX_API_KEY` | Global: [platform.minimax.io](https://platform.minimax.io) · China: [platform.minimaxi.com](https://platform.minimaxi.com) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |

MiniMax region (key and host must match):

```bash
MINIMAX_REGION=global   # platform.minimax.io (default)
# MINIMAX_REGION=cn     # platform.minimaxi.com
```

---

## Exercise 1 — Compare models

```bash
python compare_models.py
python compare_models.py --only deepseek
```

---

## Exercise 2 — Day 3–4: Short, defensible prompting

Six prompt versions of the **same** product-support task (`prompt_versions.py`), covering:

- system vs user placement
- explicit output constraints
- “I don’t know” / missing-fact handling

```bash
# 3 repeats per version (default provider: deepseek)
python prompting_lab.py

python prompting_lab.py --provider deepseek --runs 3
python prompting_lab.py --provider minimax --runs 3

# one version only
python prompting_lab.py --only v6_unknown_handling
```

Outputs land in `lab_output/`:

- `prompting_*.json` — raw answers + consistency ratio
- `prompting_notes_*.md` — fill-in ranking / interview notes

Read `DESIGN.md` for the interview soundbite.

---

## Exercise 3 — Day 5–7: Function calling basics

One tool (`calculate`) + full loop in `agent_loop.py`:

1. define tool schema  
2. parse `tool_calls`  
3. execute locally  
4. append `role=tool` result  
5. get final answer  

Same code for both providers — only `--provider` changes:

```bash
python agent_loop.py --provider deepseek
python agent_loop.py --provider minimax

python agent_loop.py --provider deepseek "What is (81 / 9) ** 2 + 3?"
python agent_loop.py --provider minimax  "What is (81 / 9) ** 2 + 3?"
```

Quick unit check of the tool (no API key):

```bash
python -c "from tools import calculate; print(calculate('(17.5 * 4) + 9 / 3'))"
```

---

## Layout

```
tool_using_agent/
  providers.py       # shared MiniMax / DeepSeek clients
  compare_models.py  # same-prompt comparison
  prompt_versions.py # 6 defensible prompt designs
  prompting_lab.py   # run + score consistency
  tools.py           # calculate schema + implementation
  agent_loop.py      # function-calling loop
  DESIGN.md          # prompting + tools design notes
  lab_output/        # run artifacts (gitignored)
```
