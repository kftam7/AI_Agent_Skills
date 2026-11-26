"""
Tool schemas + local implementations for the agent.

Day 5–7: calculate only (CALCULATE_TOOLS)
Day 8–10: calculate + convert_units + get_weather (+ optional web_search) → TOOLS
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

# --- helpers ---------------------------------------------------------------

UA = "AI-Agent-Skills-lab/1.0 (educational; contact: local)"


def _ok(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, **kwargs}


def _err(message: str, **kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **kwargs}


def _http_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


# --- calculate -------------------------------------------------------------

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
    expr = (expression or "").strip()
    if not expr:
        return _err("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
        return _ok(expression=expr, result=value)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc), expression=expr)


# --- convert_units ---------------------------------------------------------

# value_in_base = value * to_base[unit]; value_out = value_in_base / to_base[out]
_LENGTH_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}
_MASS_TO_KG = {
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
}
_TEMP = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


def _norm_unit(unit: str) -> str:
    return unit.strip().lower().replace("°", "").replace(" ", "")


def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert length, mass, or temperature units."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return _err(f"value must be a number, got {value!r}")

    src = _norm_unit(from_unit)
    dst = _norm_unit(to_unit)
    if not src or not dst:
        return _err("from_unit and to_unit are required")

    try:
        if src in _TEMP or dst in _TEMP:
            if src not in _TEMP or dst not in _TEMP:
                return _err("temperature conversion requires both units to be C/F/K")

            def to_c(u: str, v: float) -> float:
                if u in ("c", "celsius"):
                    return v
                if u in ("f", "fahrenheit"):
                    return (v - 32.0) * 5.0 / 9.0
                return v - 273.15  # kelvin

            def from_c(u: str, v: float) -> float:
                if u in ("c", "celsius"):
                    return v
                if u in ("f", "fahrenheit"):
                    return v * 9.0 / 5.0 + 32.0
                return v + 273.15

            out = from_c(dst, to_c(src, amount))
            return _ok(value=amount, from_unit=src, to_unit=dst, result=out, category="temperature")

        if src in _LENGTH_TO_M and dst in _LENGTH_TO_M:
            meters = amount * _LENGTH_TO_M[src]
            out = meters / _LENGTH_TO_M[dst]
            return _ok(value=amount, from_unit=src, to_unit=dst, result=out, category="length")

        if src in _MASS_TO_KG and dst in _MASS_TO_KG:
            kg = amount * _MASS_TO_KG[src]
            out = kg / _MASS_TO_KG[dst]
            return _ok(value=amount, from_unit=src, to_unit=dst, result=out, category="mass")

        return _err(
            f"unsupported unit pair {from_unit!r} → {to_unit!r}",
            supported="length (m/km/cm/mm/mi/ft/in), mass (kg/g/lb/oz), temp (C/F/K)",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc), value=amount, from_unit=src, to_unit=dst)


# --- get_weather (Open-Meteo, no API key) ----------------------------------

def get_weather(location: str) -> dict[str, Any]:
    """Look up current weather via Open-Meteo geocoding + forecast (no API key)."""
    place = (location or "").strip()
    if not place:
        return _err("location is required")

    try:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
        )
        geo = _http_json(geo_url)
        results = geo.get("results") or []
        if not results:
            return _err(f"location not found: {place}")

        hit = results[0]
        lat, lon = hit["latitude"], hit["longitude"]
        label = ", ".join(
            str(hit[k]) for k in ("name", "admin1", "country") if hit.get(k)
        )

        wx_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                }
            )
        )
        wx = _http_json(wx_url)
        current = wx.get("current") or {}
        return _ok(
            location=label,
            latitude=lat,
            longitude=lon,
            temperature_c=current.get("temperature_2m"),
            humidity_percent=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            weather_code=current.get("weather_code"),
            observed_at=current.get("time"),
            source="open-meteo",
        )
    except urllib.error.HTTPError as exc:
        return _err(f"weather HTTP {exc.code}: {exc.reason}", location=place)
    except urllib.error.URLError as exc:
        return _err(f"weather network error: {exc.reason}", location=place)
    except Exception as exc:  # noqa: BLE001
        return _err(f"weather lookup failed: {exc}", location=place)


# --- web_search (DuckDuckGo Instant Answer; best-effort) -------------------

def _search_duckduckgo(q: str, n: int) -> dict[str, Any]:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    data = _http_json(url)
    results: list[dict[str, str]] = []
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract:
        results.append(
            {"title": data.get("Heading") or q, "snippet": abstract, "url": abstract_url}
        )
    for item in data.get("RelatedTopics") or []:
        if len(results) >= n:
            break
        if not isinstance(item, dict) or "Topics" in item:
            continue
        text = (item.get("Text") or "").strip()
        link = (item.get("FirstURL") or "").strip()
        if text:
            results.append({"title": text[:80], "snippet": text, "url": link})
    if not results:
        return _err("no DuckDuckGo instant-answer hits", query=q, source="duckduckgo")
    return _ok(query=q, results=results[:n], source="duckduckgo-instant-answer")


def _search_wikipedia(q: str, n: int) -> dict[str, Any]:
    """Reliable no-key data lookup fallback (OpenSearch API)."""
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": q,
            "limit": n,
            "namespace": 0,
            "format": "json",
        }
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    # [query, titles[], descriptions[], urls[]]
    if not isinstance(data, list) or len(data) < 4:
        return _err("unexpected Wikipedia response shape", query=q)
    titles, descs, urls = data[1], data[2], data[3]
    results = [
        {"title": t, "snippet": d or t, "url": u}
        for t, d, u in zip(titles, descs, urls)
    ]
    if not results:
        return _err("no Wikipedia results", query=q, source="wikipedia")
    return _ok(query=q, results=results, source="wikipedia-opensearch")


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Lightweight web/data lookup. Tries DuckDuckGo Instant Answer, then Wikipedia.
    Fails gracefully if the network or APIs are unavailable.
    """
    q = (query or "").strip()
    if not q:
        return _err("query is required")
    try:
        n = max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        n = 5

    errors: list[str] = []
    for fn in (_search_duckduckgo, _search_wikipedia):
        try:
            result = fn(q, n)
            if result.get("ok"):
                return result
            errors.append(str(result.get("error") or "unknown"))
        except urllib.error.HTTPError as exc:
            errors.append(f"{fn.__name__} HTTP {exc.code}")
        except urllib.error.URLError as exc:
            errors.append(f"{fn.__name__} network: {exc.reason}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {exc}")
    return _err("; ".join(errors) or "search failed", query=q)


# --- registry + schemas ----------------------------------------------------

TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate": lambda **kwargs: calculate(str(kwargs.get("expression", ""))),
    "convert_units": lambda **kwargs: convert_units(
        kwargs.get("value"),
        str(kwargs.get("from_unit", "")),
        str(kwargs.get("to_unit", "")),
    ),
    "get_weather": lambda **kwargs: get_weather(str(kwargs.get("location", ""))),
    "web_search": lambda **kwargs: web_search(
        str(kwargs.get("query", "")),
        kwargs.get("max_results", 5),
    ),
}

CALCULATE_TOOLS: list[dict[str, Any]] = [
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

TOOLS: list[dict[str, Any]] = CALCULATE_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": (
                "Convert a numeric value between units. Supports length "
                "(m, km, cm, mm, mi, ft, in), mass (kg, g, lb, oz), and temperature (C, F, K)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Number to convert"},
                    "from_unit": {"type": "string", "description": "Source unit, e.g. 'km' or 'C'"},
                    "to_unit": {"type": "string", "description": "Target unit, e.g. 'mi' or 'F'"},
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather for a city or place name (temperature, humidity, wind). "
                "Use when the user asks about weather or outdoor conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place, e.g. 'Tokyo' or 'San Francisco, CA'",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Look up a short factual snippet / instant answer on the public web. "
                "Use for definitions or well-known facts; not a substitute for live news wire."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (1-8)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def run_tool(name: str, arguments_json: str) -> str:
    """Execute a tool by name; always return a JSON string (never raise to the model loop)."""
    if name not in TOOL_IMPLS:
        return json.dumps(_err(f"unknown tool: {name}"))
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(args, dict):
            return json.dumps(_err("arguments must be a JSON object"))
    except json.JSONDecodeError as exc:
        return json.dumps(_err(f"invalid JSON arguments: {exc}"))

    try:
        result = TOOL_IMPLS[name](**args)
    except TypeError as exc:
        # wrong/missing kwargs from the model
        result = _err(f"bad tool arguments: {exc}")
    except Exception as exc:  # noqa: BLE001
        result = _err(f"tool crashed: {type(exc).__name__}: {exc}")

    if not isinstance(result, dict):
        result = _err("tool returned a non-dict result", raw=str(result))
    return json.dumps(result, default=str)
