"""Income verification layer.

Two clearly separated responsibilities:

  LLM  (job a) — extract structured income *signals* from unstructured bank
                 statement text: salary-like recurring credits (amount, date,
                 payer), irregular/cash-like inflows, detected employer.
                 The LLM extracts; it does NOT compute or decide anything.

  Python (deterministic) — everything numeric: observed monthly income,
                 income stability (coverage + coefficient of variation),
                 declared-vs-observed variance, and the green/amber/red flags.
                 Thresholds live in underwriting.Policy.
"""

from __future__ import annotations

import statistics
from datetime import datetime

import providers
from underwriting import Flag, Policy

# --- LLM extraction (job a) --------------------------------------------------

SIGNALS_SCHEMA = {
    "type": "object",
    "properties": {
        "salary_like_credits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                    "payer": {"type": "string"},
                },
                "required": ["date", "amount", "payer"],
                "additionalProperties": False,
            },
        },
        "irregular_inflows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["date", "amount", "description"],
                "additionalProperties": False,
            },
        },
        "detected_employer": {"type": "string"},
    },
    "required": ["salary_like_credits", "irregular_inflows", "detected_employer"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You read raw Indian bank statement text and extract income signals. You do NOT "
    "compute totals, averages, or make any judgement about the applicant — only "
    "extract what is present.\n\n"
    "Return:\n"
    "- salary_like_credits: CREDIT entries that look like recurring salary/payroll — "
    "regular (roughly monthly) credits of a similar amount, often via NEFT/IMPS/RTGS "
    "with an employer/company name, or narrations containing SAL, SALARY, PAYROLL. "
    "Give amount (POSITIVE number, no commas/symbols), date (as printed), and payer "
    "(the employer/counterparty name).\n"
    "- irregular_inflows: other credits that are NOT salary-like — cash deposits, "
    "one-off transfers, uneven amounts, UPI receipts from individuals. Give amount, "
    "date, description.\n"
    "- detected_employer: the most likely employer name if identifiable, else \"\".\n\n"
    "Rules: only use CREDIT (money-in) lines; ignore debits. Do not invent entries. "
    "If unsure whether a credit is salary, put it in irregular_inflows."
)


def extract_income_signals(statement_text: str) -> dict:
    """LLM extraction -> normalized signals dict."""
    if not statement_text or not statement_text.strip():
        return {"salary_like_credits": [], "irregular_inflows": [], "detected_employer": ""}
    user = "Bank statement text:\n=== START ===\n" + statement_text + "\n=== END ==="
    raw = providers.extract_json(SYSTEM_PROMPT, user, SIGNALS_SCHEMA)
    return normalize_signals(raw)


def normalize_signals(raw: dict) -> dict:
    def _clean(items, amount_key_extra):
        out = []
        for it in items or []:
            try:
                amt = abs(float(it["amount"]))
            except (KeyError, TypeError, ValueError):
                continue
            row = {"date": str(it.get("date", "")).strip(), "amount": round(amt, 2)}
            for k in amount_key_extra:
                row[k] = str(it.get(k, "")).strip()
            out.append(row)
        return out

    return {
        "salary_like_credits": _clean(raw.get("salary_like_credits"), ["payer"]),
        "irregular_inflows": _clean(raw.get("irregular_inflows"), ["description"]),
        "detected_employer": str(raw.get("detected_employer", "")).strip(),
    }


# --- Deterministic income math (Python only) ---------------------------------

_DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
    "%d/%m/%y", "%d-%b-%Y", "%d-%b-%y", "%d.%m.%Y",
)


def _month_key(date_str: str) -> str | None:
    s = (date_str or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return None


def _last_n_months(latest: str, n: int) -> list[str]:
    year, month = (int(x) for x in latest.split("-"))
    keys = []
    for _ in range(n):
        keys.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return list(reversed(keys))


def monthly_salary_totals(salary_like_credits: list[dict]) -> dict[str, float]:
    """Sum salary-like credits per calendar month (usually one credit/month)."""
    totals: dict[str, float] = {}
    for c in salary_like_credits:
        mk = _month_key(c.get("date", ""))
        if mk is None:
            continue
        totals[mk] = totals.get(mk, 0.0) + float(c["amount"])
    return totals


def observed_monthly_income(salary_like_credits: list[dict]) -> float:
    """Median of monthly salary totals — robust to a stray double credit."""
    totals = monthly_salary_totals(salary_like_credits)
    if not totals:
        return 0.0
    return float(statistics.median(totals.values()))


def income_stability(signals: dict, policy: Policy = Policy()) -> dict:
    """Coverage and coefficient of variation of salary over the look-back window."""
    credits = signals.get("salary_like_credits", [])
    totals = monthly_salary_totals(credits)
    window = policy.stability_window_months
    if not totals:
        return {
            "window": window, "months_with_salary": 0, "coverage_ratio": 0.0,
            "cov": 0.0, "mean": 0.0, "std": 0.0, "n_salary_credits": len(credits),
        }
    keys = _last_n_months(max(totals), window)
    amounts = [totals[k] for k in keys if totals.get(k, 0.0) > 0]
    months_with_salary = len(amounts)
    mean = statistics.mean(amounts) if amounts else 0.0
    std = statistics.pstdev(amounts) if amounts else 0.0  # population std; 1 point -> 0
    cov = (std / mean) if mean > 0 else 0.0
    return {
        "window": window,
        "months_with_salary": months_with_salary,
        "coverage_ratio": months_with_salary / window,
        "cov": cov,
        "mean": mean,
        "std": std,
        "n_salary_credits": len(credits),
    }


def income_variance(declared: float, observed: float) -> float:
    """Signed variance fraction: (declared - observed) / declared. + = overstated."""
    if declared <= 0:
        return 0.0
    return (declared - observed) / declared


def basis_income(declared: float, observed: float, basis: str) -> float:
    basis = basis.strip().lower()
    if basis == "observed":
        return observed
    if basis == "lower":
        return min(declared, observed)
    return declared  # default


# --- Deterministic flags -----------------------------------------------------


def reconciliation_flag(declared: float, observed: float, policy: Policy = Policy()) -> Flag:
    overstatement = max(income_variance(declared, observed), 0.0)  # only overstatement is risk
    if overstatement <= policy.income_variance_amber:
        level = "green"
    elif overstatement <= policy.income_variance_red:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Declared vs observed income",
        level,
        f"Declared Rs {round(declared):,} vs statement-observed Rs {round(observed):,} "
        f"— declared is {overstatement * 100:.1f}% above observed "
        f"(amber > {policy.income_variance_amber * 100:.0f}%, red > "
        f"{policy.income_variance_red * 100:.0f}%).",
    )


def stability_flag(stability: dict, policy: Policy = Policy()) -> Flag:
    coverage = stability["coverage_ratio"]
    cov = stability["cov"]
    if coverage >= policy.stability_green_coverage and cov <= policy.stability_green_cov:
        level = "green"
    elif coverage >= policy.stability_amber_coverage:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Income stability (observed)",
        level,
        f"Salary credits in {stability['months_with_salary']}/{stability['window']} months "
        f"(coverage {coverage * 100:.0f}%), amount variation (CoV) {cov * 100:.1f}%.",
    )


def verify(declared_income: float, signals: dict, policy: Policy, basis: str) -> tuple[dict, list[Flag]]:
    """Full deterministic verification: returns (result_dict, extra_flags)."""
    credits = signals.get("salary_like_credits", [])
    observed = observed_monthly_income(credits)
    stability = income_stability(signals, policy)
    variance = income_variance(declared_income, observed)
    chosen = basis_income(declared_income, observed, basis)
    flags = [
        reconciliation_flag(declared_income, observed, policy),
        stability_flag(stability, policy),
    ]
    result = {
        "declared_monthly_income": declared_income,
        "observed_monthly_income": observed,
        "basis": basis,
        "basis_income": chosen,
        "variance_fraction": variance,
        "stability": stability,
        "detected_employer": signals.get("detected_employer", ""),
        "salary_like_credits": credits,
        "irregular_inflows": signals.get("irregular_inflows", []),
    }
    return result, flags
