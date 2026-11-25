"""
Tool schemas + local implementations for the Day 5–7 agent loop.

Start with one reliable tool (`calculate`). Schema is OpenAI/DeepSeek/MiniMax
compatible JSON Schema under `tools=[...]`.
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

# --- local implementations -------------------------------------------------

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("Only basic arithmetic expressions are allowed")


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a basic arithmetic expression (no names, no calls)."""
    expr = expression.strip()
    if not expr:
        return {"ok": False, "error": "empty expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
        return {"ok": True, "expression": expr, "result": value}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "expression": expr, "error": str(exc)}


TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate": lambda **kwargs: calculate(str(kwargs.get("expression", ""))),
}

# --- OpenAI-compatible tool schemas ----------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a basic arithmetic expression with + - * / // % ** and "
                "parentheses. Use for any numeric calculation instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(17.5 * 4) + 9 / 3'",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    }
]


def run_tool(name: str, arguments_json: str) -> str:
    """Execute a tool by name; always return a JSON string for the model."""
    if name not in TOOL_IMPLS:
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"})
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(args, dict):
            return json.dumps({"ok": False, "error": "arguments must be a JSON object"})
    except json.JSONDecodeError as exc:
        return json.dumps({"ok": False, "error": f"invalid JSON arguments: {exc}"})
    result = TOOL_IMPLS[name](**args)
    return json.dumps(result, default=str)
