"""Deterministic underwriting engine — the agent's tools.

ALL lending math and ALL flag/decision logic live here, in plain Python. The
LLM never computes a number or decides a flag; it only narrates the structured
output of these functions (see providers.py / app.py).

Policy thresholds (max FOIR, score bands, LTV cap) are LENDER policy, not RBI
mandates — they are configurable. RBI does publish tiered LTV caps for certain
secured loans (e.g. home loans ~90/80/75% by ticket size, gold loans ~75%);
those are noted in the README and left to the underwriter to set, rather than
hard-coded here where they could drift out of date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Policy (lender-configurable; NOT RBI-mandated) --------------------------


@dataclass
class Policy:
    max_foir: float = 0.50       # max (existing + proposed obligations) / income
    foir_green: float = 0.40     # <= green, <= max_foir amber, else red
    score_green: int = 750       # CIBIL-style band (300–900)
    score_amber: int = 650
    ltv_cap: float = 0.80        # secured-loan loan-to-value cap
    ltv_amber_band: float = 0.05  # amber within cap..cap+band, red beyond
    # Income verification (declared vs statement-observed):
    income_variance_amber: float = 0.10  # overstatement <= this -> green
    income_variance_red: float = 0.25    # overstatement <= this -> amber, else red
    stability_window_months: int = 6     # look-back window for salary credits
    stability_green_coverage: float = 0.80  # months_with_salary / window
    stability_green_cov: float = 0.10    # coefficient of variation ceiling for green
    stability_amber_coverage: float = 0.50


DEFAULT_POLICY = Policy()

LEVELS = {"green": 0, "amber": 1, "red": 2}


# --- Tools: pure financial math ----------------------------------------------


def emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Reducing-balance EMI. r = monthly rate; handles 0% cleanly."""
    if tenure_months <= 0:
        raise ValueError("tenure_months must be positive")
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)


def present_value(payment: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Loan principal supported by a monthly `payment` (inverse of emi)."""
    if tenure_months <= 0:
        raise ValueError("tenure_months must be positive")
    if payment <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return payment * tenure_months
    factor = (1 + r) ** tenure_months
    return payment * (factor - 1) / (r * factor)


def foir(existing_obligations: float, proposed_emi: float, net_monthly_income: float) -> float:
    """Fixed-Obligation-to-Income Ratio (a.k.a. DTI on net income), as a fraction."""
    if net_monthly_income <= 0:
        raise ValueError("net_monthly_income must be positive")
    return (existing_obligations + proposed_emi) / net_monthly_income


def eligible_loan_amount(
    net_monthly_income: float,
    existing_obligations: float,
    annual_rate_pct: float,
    tenure_months: int,
    max_foir: float,
) -> float:
    """Max principal that keeps FOIR within policy, given existing obligations."""
    emi_room = max_foir * net_monthly_income - existing_obligations
    if emi_room <= 0:
        return 0.0
    return present_value(emi_room, annual_rate_pct, tenure_months)


def ltv(loan_amount: float, asset_value: float | None) -> float | None:
    """Loan-to-Value as a fraction, or None for unsecured loans."""
    if not asset_value or asset_value <= 0:
        return None
    return loan_amount / asset_value


# --- Flag logic (deterministic rules, not the LLM) ---------------------------


@dataclass
class Flag:
    name: str
    level: str  # green | amber | red
    reason: str


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _rupees(x: float) -> str:
    return f"Rs {round(x):,}"


def _foir_flag(foir_val: float, p: Policy) -> Flag:
    if foir_val <= p.foir_green:
        level = "green"
    elif foir_val <= p.max_foir:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Repayment capacity (FOIR/DTI)",
        level,
        f"FOIR is {_pct(foir_val)} including the proposed EMI; policy cap is "
        f"{_pct(p.max_foir)} (comfort {_pct(p.foir_green)}).",
    )


def _eligibility_flag(requested: float, eligible: float) -> Flag:
    if eligible <= 0:
        return Flag(
            "Loan eligibility",
            "red",
            "Eligible amount is Rs 0 — existing obligations already exhaust the FOIR budget.",
        )
    ratio = requested / eligible
    if ratio <= 0.9:
        level = "green"
    elif ratio <= 1.0:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Loan eligibility",
        level,
        f"Requested {_rupees(requested)} vs policy-eligible {_rupees(eligible)} "
        f"({_pct(ratio)} of eligibility).",
    )


def _bureau_flag(score: int, p: Policy) -> Flag:
    if score is None or score <= 0:
        return Flag(
            "Credit bureau score",
            "amber",
            "No bureau history (new-to-credit) — thin file, manual verification needed.",
        )
    if score >= p.score_green:
        level = "green"
    elif score >= p.score_amber:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Credit bureau score",
        level,
        f"Bureau score {score} (green >= {p.score_green}, amber >= {p.score_amber}).",
    )


def _ltv_flag(ltv_val: float, p: Policy) -> Flag:
    if ltv_val <= p.ltv_cap:
        level = "green"
    elif ltv_val <= p.ltv_cap + p.ltv_amber_band:
        level = "amber"
    else:
        level = "red"
    return Flag(
        "Loan-to-Value (LTV)",
        level,
        f"LTV is {_pct(ltv_val)}; secured-loan cap is {_pct(p.ltv_cap)}.",
    )


def _employment_flag(employment_type: str) -> Flag:
    if employment_type.strip().lower().startswith("self"):
        return Flag(
            "Income stability",
            "amber",
            "Self-employed income — verify ITR/GST filings and average bank balances.",
        )
    return Flag(
        "Income stability",
        "green",
        "Salaried applicant — income treated as stable (verify salary credits).",
    )


# --- Orchestration -----------------------------------------------------------


def _worst(flags: list[Flag]) -> str:
    return max((f.level for f in flags), key=lambda lv: LEVELS[lv])


def _recommendation(level: str) -> dict:
    return {
        "green": {"decision": "Recommend approval", "level": "green"},
        "amber": {"decision": "Refer for manual review (conditional approval possible)", "level": "amber"},
        "red": {"decision": "Decline / escalate to credit committee", "level": "red"},
    }[level]


def assess(
    applicant: dict,
    policy: Policy = DEFAULT_POLICY,
    extra_flags: list[Flag] | None = None,
) -> dict:
    """Run all tools and rules; return a fully deterministic assessment dict.

    `extra_flags` (e.g. income-verification flags computed in income.py) are
    folded into the flag set and the overall recommendation — the final decision
    still comes only from deterministic Python rules, never the LLM.
    """
    income = float(applicant["net_monthly_income"])
    existing = float(applicant.get("existing_obligations", 0) or 0)
    requested = float(applicant["loan_amount_requested"])
    tenure = int(applicant["tenure_months"])
    rate = float(applicant["interest_rate_pct"])
    score = applicant.get("bureau_score")
    score = int(score) if score is not None else None
    employment = str(applicant.get("employment_type", "Salaried"))
    asset_value = applicant.get("asset_value")
    asset_value = float(asset_value) if asset_value else None

    proposed_emi = emi(requested, rate, tenure)
    foir_val = foir(existing, proposed_emi, income)
    eligible = eligible_loan_amount(income, existing, rate, tenure, policy.max_foir)
    ltv_val = ltv(requested, asset_value)
    emi_room = max(policy.max_foir * income - existing, 0.0)

    flags = [
        _foir_flag(foir_val, policy),
        _eligibility_flag(requested, eligible),
        _bureau_flag(score, policy),
        _employment_flag(employment),
    ]
    if ltv_val is not None:
        flags.append(_ltv_flag(ltv_val, policy))
    if extra_flags:
        flags.extend(extra_flags)

    level = _worst(flags)
    return {
        "inputs": {
            "net_monthly_income": income,
            "existing_obligations": existing,
            "loan_amount_requested": requested,
            "tenure_months": tenure,
            "interest_rate_pct": rate,
            "bureau_score": score,
            "employment_type": employment,
            "asset_value": asset_value,
        },
        "metrics": {
            "proposed_emi": round(proposed_emi, 2),
            "foir": round(foir_val, 4),
            "dti": round(foir_val, 4),  # DTI on net income == FOIR here
            "eligible_loan_amount": round(eligible, 2),
            "max_affordable_emi": round(emi_room, 2),
            "ltv": round(ltv_val, 4) if ltv_val is not None else None,
        },
        "flags": [f.__dict__ for f in flags],
        "recommendation": _recommendation(level),
    }
