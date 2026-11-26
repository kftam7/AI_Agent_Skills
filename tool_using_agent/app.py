#!/usr/bin/env python3
"""
Day 11–14: Streamlit demo UI for the multi-tool working agent.

Run:
  streamlit run app.py --server.port 8505
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from providers import load_agent_config
from working_agent import run_working_agent

st.set_page_config(
    page_title="AI Agent Skills — Tool Agent",
    page_icon="🛠️",
    layout="centered",
)

cfg = load_agent_config()

st.title("Tool-using agent demo")
st.caption(
    "Days 8–14 lab: calculate · convert_units · get_weather · web_search. "
    "Provider switch is config-only — same agent loop."
)

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "Provider",
        options=["deepseek", "minimax", "grok"],
        index=["deepseek", "minimax", "grok"].index(
            cfg.get("provider", "deepseek")
            if cfg.get("provider") in ("deepseek", "minimax", "grok")
            else "deepseek"
        ),
        help="Requires the matching API key in ../.env",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.get("temperature", 0.2)),
        step=0.05,
    )
    max_steps = st.number_input(
        "Max loop steps",
        min_value=1,
        max_value=16,
        value=int(cfg.get("max_steps", 8)),
    )
    st.markdown("---")
    st.markdown(
        "**Tips**\n"
        "- Weather + units in one question exercises multi-tool calls.\n"
        "- Tool failures return `ok: false` — the agent should not invent facts.\n"
        "- Full JSON logs land in `lab_output/conversations/`."
    )

examples = [
    "What's the weather in Tokyo in Fahrenheit, and how many miles is 12 km?",
    "Calculate (81 / 9) ** 2 + 3, then convert 100 C to F.",
    "Look up what Open-Meteo is, in one sentence.",
]
pick = st.selectbox("Example questions", options=["(type your own)"] + examples)
default_q = "" if pick == "(type your own)" else pick

question = st.text_area("Your question", value=default_q, height=100)

col1, col2 = st.columns([1, 1])
run_clicked = col1.button("Run agent", type="primary", use_container_width=True)
clear_clicked = col2.button("Clear result", use_container_width=True)

if clear_clicked:
    for key in ("last_answer", "last_log", "last_trace", "last_error"):
        st.session_state.pop(key, None)
    st.rerun()

if run_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner(f"Running agent via {provider}…"):
            try:
                answer, log_path = run_working_agent(
                    provider,
                    question.strip(),
                    max_steps=int(max_steps),
                    temperature=float(temperature),
                    verbose=False,
                )
                payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
                st.session_state["last_answer"] = answer
                st.session_state["last_log"] = str(log_path)
                st.session_state["last_trace"] = payload.get("trace", [])
                st.session_state["last_error"] = None
            except SystemExit as exc:
                st.session_state["last_error"] = str(exc) or "Missing API key or bad config."
                st.session_state.pop("last_answer", None)
            except Exception as exc:  # noqa: BLE001
                st.session_state["last_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state.pop("last_answer", None)

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

if st.session_state.get("last_answer") is not None:
    st.subheader("Final answer")
    st.write(st.session_state["last_answer"])

    st.subheader("Tool / loop trace")
    trace = st.session_state.get("last_trace") or []
    if not trace:
        st.info("No trace recorded.")
    else:
        for event in trace:
            kind = event.get("event")
            step = event.get("step")
            if kind == "assistant":
                calls = event.get("tool_calls") or []
                with st.expander(f"Step {step} · assistant · finish={event.get('finish_reason')}"):
                    if event.get("content"):
                        st.markdown(event["content"])
                    if calls:
                        st.code(json.dumps(calls, indent=2), language="json")
            elif kind == "tool_result":
                ok = (event.get("result") or {}).get("ok")
                label = "OK" if ok else "ERR"
                with st.expander(f"Step {step} · tool [{label}] · {event.get('name')}"):
                    st.code(event.get("arguments") or "{}", language="json")
                    st.json(event.get("result") or {})
            elif kind == "model_error":
                st.error(f"Step {step} model error: {event.get('error')}")

    log_path = st.session_state.get("last_log")
    if log_path:
        st.caption(f"Saved conversation log: `{log_path}`")
