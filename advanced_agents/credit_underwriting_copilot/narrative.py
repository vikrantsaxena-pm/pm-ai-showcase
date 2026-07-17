"""Turn the deterministic assessment into an underwriter narrative.

The LLM is given the already-computed numbers, flags, and recommendation and
asked ONLY to explain them in prose. It is explicitly forbidden from changing
any number or the decision. If no LLM/key is configured, a deterministic
template narrative is produced instead, so the app is fully usable offline.
"""

from __future__ import annotations

import json

import providers

SYSTEM_PROMPT = (
    "You are a credit underwriting copilot. You are given a JSON assessment that was "
    "already computed by deterministic tools: financial metrics, red/amber/green flags "
    "with reasons, and a final recommendation.\n\n"
    "Write a concise underwriter narrative (120-200 words) that explains the "
    "recommendation to a human credit officer.\n\n"
    "Hard rules:\n"
    "- Use ONLY the numbers, flags, and recommendation provided. Do NOT compute, "
    "re-derive, or alter any figure.\n"
    "- Do NOT change or second-guess the recommendation decision; restate it as given.\n"
    "- Walk through the material flags (worst first) and what each means for risk.\n"
    "- If a flag is amber, state the condition/verification that would clear it.\n"
    "- If 'income_verification' is present, note the declared vs statement-observed "
    "income, which income basis the assessment used, and what the salary credits show. "
    "Do not recompute any of it.\n"
    "- Plain professional English. No markdown headers. Rupee amounts as 'Rs'."
)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _rupees(x: float) -> str:
    return f"Rs {round(x):,}"


def fallback_narrative(a: dict) -> str:
    """Deterministic narrative used when no LLM is configured."""
    m, rec = a["metrics"], a["recommendation"]
    order = {"red": 0, "amber": 1, "green": 2}
    flags = sorted(a["flags"], key=lambda f: order[f["level"]])
    lines = [
        f"Recommendation: {rec['decision']}.",
        (
            f"On a requested {_rupees(a['inputs']['loan_amount_requested'])} over "
            f"{a['inputs']['tenure_months']} months at {a['inputs']['interest_rate_pct']}%, "
            f"the proposed EMI is {_rupees(m['proposed_emi'])}, taking FOIR/DTI to "
            f"{_pct(m['foir'])}. Policy-eligible amount is {_rupees(m['eligible_loan_amount'])}."
        ),
    ]
    if m["ltv"] is not None:
        lines.append(f"LTV on the pledged asset is {_pct(m['ltv'])}.")
    iv = a.get("income_verification")
    if iv:
        lines.append(
            f"Income verification: declared {_rupees(iv['declared_monthly_income'])} vs "
            f"statement-observed {_rupees(iv['observed_monthly_income'])}; assessed on the "
            f"'{iv['basis']}' basis ({_rupees(iv['basis_income'])}). Salary credits seen in "
            f"{iv['stability']['months_with_salary']}/{iv['stability']['window']} months."
        )
    lines.append("Key factors:")
    for f in flags:
        lines.append(f"- [{f['level'].upper()}] {f['name']}: {f['reason']}")
    return "\n".join(lines)


def compose(assessment: dict) -> tuple[str, str]:
    """Return (narrative_text, source) where source is 'llm' or 'fallback'."""
    if not providers.has_api_key():
        return fallback_narrative(assessment), "fallback"
    user = "Assessment JSON:\n" + json.dumps(assessment, indent=2)
    try:
        return providers.generate(SYSTEM_PROMPT, user), "llm"
    except Exception:  # noqa: BLE001 - any LLM/network failure -> deterministic fallback
        return fallback_narrative(assessment), "fallback"
