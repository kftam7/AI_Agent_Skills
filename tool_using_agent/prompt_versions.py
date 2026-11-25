"""
Six prompt designs for the SAME task — Day 3–4 lab.

Task: answer a short factual question about a product from a tiny knowledge
snippet. Good for interviewing “why this format” because you can defend
role / rules / output / unknown handling separately.
"""

from __future__ import annotations

# Shared user-facing task content (the “same task”)
KNOWLEDGE = (
    "Product: Aurora Bottle\n"
    "Capacity: 750 ml\n"
    "Material: stainless steel\n"
    "Warranty: 2 years\n"
    "Colors in stock: black, blue\n"
    "NOT listed: dishwasher safety, price, weight"
)

QUESTION = "Is the Aurora Bottle dishwasher-safe, and what colors are in stock?"

# Each version: id, interview rationale, optional system, user message template
VERSIONS: list[dict] = [
    {
        "id": "v1_vague_user_only",
        "focus": "baseline — no role, no structure",
        "defend": "Shows what happens with an unstructured ask; often verbose and invents facts.",
        "system": None,
        "user": (
            f"Here is some info:\n{KNOWLEDGE}\n\n"
            f"Question: {QUESTION}\n"
            "Please answer."
        ),
    },
    {
        "id": "v2_role_in_user",
        "focus": "role stuffed into the user message",
        "defend": "Role helps, but mixing role+task in one blob is harder to reuse and A/B test.",
        "system": None,
        "user": (
            "You are a careful product-support assistant. "
            "Only use the product info below.\n\n"
            f"{KNOWLEDGE}\n\n"
            f"Question: {QUESTION}"
        ),
    },
    {
        "id": "v3_system_role_loose_user",
        "focus": "system = role; user = task only",
        "defend": "Separates stable policy (system) from changing task (user) — easier to defend and reuse.",
        "system": (
            "You are a careful product-support assistant. "
            "Prefer short answers. Use only the product info the user provides."
        ),
        "user": f"Product info:\n{KNOWLEDGE}\n\nQuestion: {QUESTION}",
    },
    {
        "id": "v4_structured_system",
        "focus": "short structured system: role + rules + format",
        "defend": "Interview-friendly: each line is a testable rule (role / grounding / format).",
        "system": (
            "Role: product-support assistant.\n"
            "Rules:\n"
            "1) Use ONLY the product info in the user message.\n"
            "2) Be concise.\n"
            "Output format:\n"
            "- Answer: <one or two sentences>\n"
            "- Sources used: <field names from the info>"
        ),
        "user": f"Product info:\n{KNOWLEDGE}\n\nQuestion: {QUESTION}",
    },
    {
        "id": "v5_explicit_constraints",
        "focus": "rigid output constraints (machine-checkable)",
        "defend": "Constraints make consistency measurable — you can grep/parse the shape.",
        "system": (
            "Role: product-support assistant.\n"
            "Rules: use only the provided product info; no marketing fluff.\n"
            "Output EXACTLY these two lines and nothing else:\n"
            "dishwasher_safe: yes|no|unknown\n"
            "colors_in_stock: <comma-separated colors or unknown>"
        ),
        "user": f"Product info:\n{KNOWLEDGE}\n\nQuestion: {QUESTION}",
    },
    {
        "id": "v6_unknown_handling",
        "focus": "explicit I-don't-know policy + format",
        "defend": "Shows you designed for failure modes (missing facts) instead of hoping the model refuses.",
        "system": (
            "Role: product-support assistant.\n"
            "Rules:\n"
            "1) Use ONLY the product info in the user message.\n"
            "2) If a fact is missing, say \"I don't know\" — do NOT guess.\n"
            "3) Still answer any parts that ARE in the info.\n"
            "Output format (exactly):\n"
            "known: <facts you can support>\n"
            "unknown: <facts not in the info, or 'none'>"
        ),
        "user": f"Product info:\n{KNOWLEDGE}\n\nQuestion: {QUESTION}",
    },
]
