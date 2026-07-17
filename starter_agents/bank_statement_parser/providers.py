"""Provider-agnostic transaction extraction.

Reads the LLM provider, model, and API key from the environment so the app is
never tied to one vendor. Supports Claude (Anthropic), OpenAI, and Gemini
(Google). Each backend is asked to return the same JSON shape, so the rest of
the app doesn't care which one is configured.

Env vars:
  LLM_PROVIDER   claude | openai | gemini      (default: claude)
  LLM_MODEL      overrides the per-provider default model
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import json
import os

# Categories the model must choose from. Tuned for Indian retail banking, where
# UPI / IMPS / NEFT transfers and ATM withdrawals dominate typical statements.
CATEGORIES = [
    "Income",
    "Food & Dining",
    "Groceries",
    "Transport",
    "Bills & Utilities",
    "Shopping",
    "Entertainment",
    "Transfers",
    "Cash & ATM",
    "Investments",
    "Fees & Charges",
    "Health",
    "Education",
    "Other",
]

DEFAULT_MODELS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}

SYSTEM_PROMPT = (
    "You are a meticulous financial statement parser for Indian bank accounts. "
    "You are given the raw text of a bank or UPI account statement. Extract every "
    "individual transaction line. For each transaction return:\n"
    "  - date: the transaction date exactly as printed (keep the statement's format)\n"
    "  - description: the merchant / counterparty / narration, cleaned of noise\n"
    "  - amount: the transaction value as a POSITIVE number, no currency symbol or commas\n"
    "  - direction: 'in' for money received (credit/deposit) or 'out' for money spent (debit/withdrawal)\n"
    "  - category: exactly one of the allowed categories\n\n"
    "Rules:\n"
    "  - Salary, refunds, interest credited, and incoming transfers are 'in'.\n"
    "  - UPI/IMPS/NEFT/RTGS payments to people or 'self' transfers are category 'Transfers'.\n"
    "  - ATM withdrawals are 'Cash & ATM'. Bank charges, GST, and penalties are 'Fees & Charges'.\n"
    "  - Swiggy/Zomato/restaurants are 'Food & Dining'; BigBasket/Blinkit/kirana are 'Groceries'.\n"
    "  - Uber/Ola/fuel/metro/IRCTC are 'Transport'. Electricity/mobile/DTH/rent are 'Bills & Utilities'.\n"
    "  - SIP/mutual funds/stocks/FD are 'Investments'. If genuinely unclear, use 'Other'.\n"
    "  - Do NOT invent transactions. Only extract what is present in the text.\n"
    "  - Do NOT include opening/closing balance lines as transactions.\n"
    f"Allowed categories: {', '.join(CATEGORIES)}."
)

# JSON Schema shared by the providers that support structured output.
_SCHEMA = {
    "type": "object",
    "properties": {
        "transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                    "direction": {"type": "string", "enum": ["in", "out"]},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["date", "description", "amount", "direction", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["transactions"],
    "additionalProperties": False,
}


class ProviderError(RuntimeError):
    """Raised for configuration or API problems, with a user-friendly message."""


def active_provider() -> str:
    return os.getenv("LLM_PROVIDER", "claude").strip().lower()


def active_model() -> str:
    provider = active_provider()
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider, "")


def _require_key(*names: str) -> str:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    raise ProviderError(
        f"Missing API key. Set one of: {', '.join(names)} in your environment (.env)."
    )


def _user_prompt(statement_text: str) -> str:
    return (
        "Here is the bank statement text. Extract all transactions as JSON matching "
        'the schema {"transactions": [...]}.\n\n=== STATEMENT START ===\n'
        + statement_text
        + "\n=== STATEMENT END ==="
    )


def extract_transactions(statement_text: str) -> list[dict]:
    """Return a list of transaction dicts for the configured provider."""
    provider = active_provider()
    model = active_model()
    if provider in ("claude", "anthropic"):
        raw = _extract_claude(model, statement_text)
    elif provider == "openai":
        raw = _extract_openai(model, statement_text)
    elif provider in ("gemini", "google"):
        raw = _extract_gemini(model, statement_text)
    else:
        raise ProviderError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: claude, openai, gemini."
        )
    return _normalize(raw)


# --- Claude (Anthropic) ------------------------------------------------------
def _extract_claude(model: str, statement_text: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(statement_text)}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


# --- OpenAI ------------------------------------------------------------------
def _extract_openai(model: str, statement_text: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(statement_text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "transactions", "schema": _SCHEMA, "strict": True},
        },
    )
    return json.loads(resp.choices[0].message.content)


# --- Gemini (Google) ---------------------------------------------------------
def _extract_gemini(model: str, statement_text: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=_user_prompt(statement_text),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SCHEMA,
        ),
    )
    return json.loads(resp.text)


# --- Normalization -----------------------------------------------------------
def _parse_amount(value) -> float | None:
    """Parse an amount that may arrive as a number or a messy string.

    Indian statements commonly print grouped amounts like ``85,000.00`` or
    ``₹ 1,20,500`` (lakh grouping), so strip currency symbols, commas, and
    spaces before converting. Returns the absolute value, or None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return abs(float(value))
    cleaned = (
        str(value)
        .replace(",", "")
        .replace(RUPEE, "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .replace("INR", "")
        .strip()
    )
    try:
        return abs(float(cleaned))
    except (TypeError, ValueError):
        return None


RUPEE = "₹"


def cashflow_summary(transactions: list[dict]) -> dict:
    """Deterministic cashflow rollup from normalized transactions.

    Pure Python (no pandas/LLM) so it's unit-testable: total in/out, net, count,
    and spending-by-category (outflows only, largest first).
    """
    total_in = sum(t["amount"] for t in transactions if t["direction"] == "in")
    total_out = sum(t["amount"] for t in transactions if t["direction"] == "out")
    by_category: dict[str, float] = {}
    for t in transactions:
        if t["direction"] == "out":
            by_category[t["category"]] = by_category.get(t["category"], 0.0) + t["amount"]
    return {
        "total_in": round(total_in, 2),
        "total_out": round(total_out, 2),
        "net": round(total_in - total_out, 2),
        "count": len(transactions),
        "by_category": {
            k: round(v, 2)
            for k, v in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
        },
    }


def _normalize(raw: dict) -> list[dict]:
    txns = raw.get("transactions", []) if isinstance(raw, dict) else []
    cleaned: list[dict] = []
    for t in txns:
        amount = _parse_amount(t.get("amount"))
        if amount is None:
            continue
        direction = str(t.get("direction", "out")).lower()
        if direction not in ("in", "out"):
            direction = "in" if amount >= 0 else "out"
        category = t.get("category") or "Other"
        if category not in CATEGORIES:
            category = "Other"
        cleaned.append(
            {
                "date": str(t.get("date", "")).strip(),
                "description": str(t.get("description", "")).strip(),
                "amount": round(amount, 2),
                "direction": direction,
                "category": category,
            }
        )
    return cleaned
