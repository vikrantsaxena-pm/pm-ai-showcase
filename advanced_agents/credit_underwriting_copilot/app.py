"""Credit Underwriting Copilot — Streamlit UI.

Pipeline:
  1. Underwriter enters declared applicant data.
  2. (Optional) Income verification: upload/paste a bank statement -> the LLM
     extracts structured income signals (salary-like credits, irregular inflows,
     employer). Python then reconciles declared vs observed income, computes
     stability, and flags variance. The extraction is shown side-by-side with the
     declared figures so it's auditable, not trusted blindly.
  3. Deterministic Python tools compute EMI/FOIR/eligibility/LTV on the chosen
     income basis (declared / observed / lower) and assign red/amber/green flags.
  4. The LLM writes the narrative from the computed JSON.

The LLM only (a) extracts signals from text and (b) writes prose. All math,
thresholds, flags, and the recommendation are deterministic Python.
"""

from __future__ import annotations

import html
import json

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import income
import narrative
import providers
from underwriting import Policy, assess

load_dotenv()

st.set_page_config(page_title="Credit Underwriting Copilot", page_icon="🏦", layout="wide")

LEVEL_COLOR = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}
LEVEL_BG = {"green": "#ecfdf5", "amber": "#fffbeb", "red": "#fef2f2"}
LEVEL_ICON = {"green": "✅", "amber": "⚠️", "red": "⛔"}
EMPLOYMENT_OPTIONS = ["Salaried", "Self-employed"]
BASIS_LABELS = {"Observed (verified)": "observed", "Lower of the two": "lower", "Declared": "declared"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .banner {padding: 1.1rem 1.4rem; border-radius: 12px; color: #fff;
               font-size: 1.35rem; font-weight: 700; margin: .3rem 0 1rem 0;}
      .flag-card {border-left: 6px solid #ccc; border-radius: 8px; padding: .7rem 1rem;
                  margin-bottom: .6rem; background: #fff;}
      .flag-title {font-weight: 700; font-size: 1rem;}
      .flag-reason {color: #444; font-size: .92rem; margin-top: .15rem;}
      .pill {display:inline-block; padding: .1rem .55rem; border-radius: 999px;
             font-size:.72rem; font-weight:700; color:#fff; margin-left:.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def inr(x) -> str:
    n = round(float(x))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3, rest, parts = s[-3:], s[:-3], []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"{sign}₹{','.join(parts)},{last3}"


def read_pdf_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()


def html_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a small table as plain HTML (avoids the pandas/pyarrow render path)."""
    head = "".join(
        f"<th style='text-align:left;padding:6px 10px;border-bottom:2px solid #ddd'>{h}</th>"
        for _, h in columns
    )
    body = ""
    for r in rows:
        cells = "".join(
            f"<td style='padding:6px 10px;border-bottom:1px solid #eee'>"
            f"{html.escape(str(r.get(k, '')))}</td>"
            for k, _ in columns
        )
        body += f"<tr>{cells}</tr>"
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:.92rem'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


# --- Input state -------------------------------------------------------------
DEFAULTS = {
    "name": "", "income": 75000, "existing": 8000, "score": 740,
    "employment": "Salaried", "requested": 800000, "tenure": 48,
    "rate": 12.0, "asset": 0, "purpose": "Personal loan (unsecured)",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"in_{k}", v)
st.session_state.setdefault("income_signals", None)
st.session_state.setdefault("stmt_text", "")


def _apply_applicant(a: dict) -> None:
    emp = str(a.get("employment_type", "Salaried"))
    emp = "Self-employed" if emp.strip().lower().startswith("self") else "Salaried"
    st.session_state.update({
        "in_name": a.get("applicant_name", ""),
        "in_income": int(a.get("net_monthly_income", 0)),
        "in_existing": int(a.get("existing_obligations", 0) or 0),
        "in_score": int(a.get("bureau_score") or 0),
        "in_employment": emp,
        "in_requested": int(a.get("loan_amount_requested", 0)),
        "in_tenure": int(a.get("tenure_months", 12)),
        "in_rate": float(a.get("interest_rate_pct", 12.0)),
        "in_asset": int(a.get("asset_value") or 0),
        "in_purpose": a.get("loan_purpose", ""),
    })


# --- Button callbacks --------------------------------------------------------
# Widget-keyed session state (in_*, stmt_text) must be written in on_click
# callbacks, which run BEFORE the widgets are re-instantiated on the next run.
# Writing them inline in the script body after the widget exists raises
# StreamlitAPIException.
def _cb_load_sample_applicant() -> None:
    with open("sample_applicant.json") as fh:
        _apply_applicant(json.load(fh))


def _cb_apply_uploaded_applicant() -> None:
    up = st.session_state.get("applicant_json_upload")
    if up is None:
        return
    try:
        up.seek(0)
        _apply_applicant(json.load(up))
    except Exception as exc:  # noqa: BLE001 - surface via a flash message after rerun
        st.session_state["_flash"] = ("error", f"Couldn't read JSON: {exc}")


def _cb_load_sample_statement() -> None:
    with open("sample_statement.txt") as fh:
        st.session_state["stmt_text"] = fh.read()


def _cb_use_sample_extraction() -> None:
    with open("sample_statement_signals.json") as fh:
        st.session_state["income_signals"] = income.normalize_signals(json.load(fh))


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**LLM:** `{providers.llm_provider()}` / `{providers.llm_model() or 'not set'}`")
    st.caption(
        "LLM does two jobs only: extract income signals from statement text, and "
        "write the narrative. All numbers and flags are deterministic Python."
    )
    st.divider()
    st.subheader("Lending policy")
    st.caption("Lender-set thresholds (not RBI mandates).")
    max_foir = st.slider("Max FOIR / DTI", 0.30, 0.70, 0.50, 0.01)
    foir_green = st.slider("FOIR comfort (green ≤)", 0.20, max_foir, min(0.40, max_foir), 0.01)
    score_green = st.slider("Bureau green (≥)", 700, 850, 750, 5)
    score_amber = st.slider("Bureau amber (≥)", 550, score_green, 650, 5)
    ltv_cap = st.slider("Secured LTV cap", 0.50, 0.95, 0.80, 0.01)
    st.subheader("Income verification")
    var_amber = st.slider("Income variance amber (>)", 0.05, 0.30, 0.10, 0.01)
    var_red = st.slider("Income variance red (>)", var_amber, 0.60, max(0.25, var_amber), 0.01)
    stab_window = st.slider("Stability window (months)", 3, 12, 6, 1)

policy = Policy(
    max_foir=max_foir, foir_green=foir_green, score_green=score_green,
    score_amber=score_amber, ltv_cap=ltv_cap, income_variance_amber=var_amber,
    income_variance_red=var_red, stability_window_months=stab_window,
)

# --- Header ------------------------------------------------------------------
st.title("🏦 Credit Underwriting Copilot")
st.caption(
    "Deterministic lending math + income verification against a bank statement. "
    "The LLM reads the statement and writes the narrative — it never computes or decides."
)

# --- Load applicant ----------------------------------------------------------
c1, c2 = st.columns([1, 2])
with c1:
    st.button(
        "📄 Load sample applicant", on_click=_cb_load_sample_applicant, use_container_width=True
    )
with c2:
    up = st.file_uploader(
        "…or upload applicant JSON", type=["json"],
        key="applicant_json_upload", label_visibility="collapsed",
    )
    st.button(
        "Apply uploaded file", on_click=_cb_apply_uploaded_applicant,
        disabled=up is None, use_container_width=True,
    )
if "_flash" in st.session_state:
    level, msg = st.session_state.pop("_flash")
    getattr(st, level)(msg)

# --- Applicant details -------------------------------------------------------
st.subheader("1. Applicant details (declared)")
st.text_input("Applicant name / ref (optional)", key="in_name")
a, b, c = st.columns(3)
with a:
    st.number_input("Net monthly income — DECLARED (₹)", min_value=0, step=1000, key="in_income")
    st.number_input("Loan amount requested (₹)", min_value=0, step=10000, key="in_requested")
    st.number_input("Asset value (₹, 0 = unsecured)", min_value=0, step=10000, key="in_asset")
with b:
    st.number_input("Existing monthly obligations (₹)", min_value=0, step=500, key="in_existing")
    st.number_input("Tenure (months)", min_value=1, max_value=360, step=6, key="in_tenure")
    st.number_input("Interest rate (% p.a.)", min_value=0.0, max_value=40.0, step=0.25, key="in_rate")
with c:
    st.number_input("Bureau score (300–900, 0 = new-to-credit)", min_value=0, max_value=900, step=5, key="in_score")
    st.selectbox("Employment type", EMPLOYMENT_OPTIONS, key="in_employment")
    st.text_input("Loan purpose (optional)", key="in_purpose")

declared_income = int(st.session_state.in_income)

# --- Income verification -----------------------------------------------------
st.subheader("2. Income verification (optional)")
st.caption(
    "Upload/paste a bank statement. The LLM extracts salary-like credits; Python "
    "reconciles them against the declared income above."
)
stmt_pdf = st.file_uploader("Bank statement PDF", type=["pdf"], key="stmt_pdf")
st.text_area("…or paste statement text", key="stmt_text", height=140)

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    extract_clicked = st.button("🔎 Extract income signals", use_container_width=True)
with bcol2:
    st.button(
        "Load sample statement text", on_click=_cb_load_sample_statement,
        use_container_width=True,
    )
with bcol3:
    st.button(
        "Use sample extraction (offline)", on_click=_cb_use_sample_extraction,
        use_container_width=True,
    )

if extract_clicked:
    text = read_pdf_text(stmt_pdf) if stmt_pdf is not None else st.session_state.stmt_text
    if not (text or "").strip():
        st.warning("Provide a statement PDF or paste statement text first.")
    else:
        with st.spinner(f"Extracting income signals with {providers.llm_provider()}…"):
            try:
                st.session_state.income_signals = income.extract_income_signals(text)
                st.success("Extracted income signals from the statement.")
            except providers.ProviderError as exc:
                st.error(
                    f"{exc}\n\nNo LLM key? Click **Use sample extraction (offline)** to "
                    "demo the reconciliation with the bundled fixture."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Extraction failed: {exc}")

signals = st.session_state.income_signals
income_result, income_flags, basis = None, None, "declared"

if signals is not None:
    observed = income.observed_monthly_income(signals.get("salary_like_credits", []))
    stability = income.income_stability(signals, policy)

    st.markdown("#### Extracted vs declared (audit)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Declared income", inr(declared_income))
    a2.metric("Observed (statement)", inr(observed))
    var = income.income_variance(declared_income, observed)
    a3.metric("Variance", f"{var * 100:+.1f}%", help="(declared − observed) / declared")
    a4.metric("Salary months", f"{stability['months_with_salary']}/{stability['window']}")
    if signals.get("detected_employer"):
        st.caption(f"Detected employer: **{signals['detected_employer']}**  ·  "
                   f"amount variation (CoV): {stability['cov'] * 100:.1f}%")

    creds = signals.get("salary_like_credits", [])
    if creds:
        st.markdown("**Salary-like credits (LLM-extracted — verify before trusting):**")
        rows = [{"date": x["date"], "payer": x.get("payer", ""), "amount": inr(x["amount"])}
                for x in creds]
        st.markdown(
            html_table(rows, [("date", "Date"), ("payer", "Payer"), ("amount", "Amount")]),
            unsafe_allow_html=True,
        )
    irr = signals.get("irregular_inflows", [])
    if irr:
        with st.expander(f"Irregular / cash-like inflows ({len(irr)}) — excluded from income"):
            rows = [{"date": x["date"], "desc": x.get("description", ""), "amount": inr(x["amount"])}
                    for x in irr]
            st.markdown(
                html_table(rows, [("date", "Date"), ("desc", "Description"), ("amount", "Amount")]),
                unsafe_allow_html=True,
            )

    basis_label = st.radio(
        "Assess on which income basis?", list(BASIS_LABELS.keys()),
        index=1, horizontal=True,
    )
    basis = BASIS_LABELS[basis_label]
    income_result, income_flags = income.verify(declared_income, signals, policy, basis)
    st.info(
        f"Assessing on **{basis_label}** = {inr(income_result['basis_income'])} "
        f"(declared {inr(declared_income)} · observed {inr(observed)})."
    )

# --- Assess ------------------------------------------------------------------
st.subheader("3. Assess")
if not st.button("Assess applicant", type="primary", use_container_width=True):
    st.info("Fill in details (and optionally verify income), then click **Assess applicant**.")
    st.stop()

assess_income = income_result["basis_income"] if income_result else declared_income
applicant = {
    "applicant_name": st.session_state.in_name,
    "net_monthly_income": assess_income,
    "existing_obligations": st.session_state.in_existing,
    "bureau_score": st.session_state.in_score or None,
    "employment_type": st.session_state.in_employment,
    "loan_amount_requested": st.session_state.in_requested,
    "tenure_months": st.session_state.in_tenure,
    "interest_rate_pct": st.session_state.in_rate,
    "asset_value": st.session_state.in_asset or None,
    "loan_purpose": st.session_state.in_purpose,
}

try:
    result = assess(applicant, policy, extra_flags=income_flags)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not assess: {exc}")
    st.stop()

if income_result:
    result["income_verification"] = income_result

rec, m = result["recommendation"], result["metrics"]

st.markdown(
    f"<div class='banner' style='background:{LEVEL_COLOR[rec['level']]}'>"
    f"{LEVEL_ICON[rec['level']]} {rec['decision']}</div>",
    unsafe_allow_html=True,
)
if income_result:
    st.caption(
        f"Assessed on **{basis}** income = {inr(assess_income)} "
        f"(declared {inr(declared_income)}, observed {inr(income_result['observed_monthly_income'])})."
    )

st.subheader("Computed metrics")
cols = st.columns(4)
cols[0].metric("Proposed EMI", inr(m["proposed_emi"]))
cols[1].metric("FOIR / DTI", f"{m['foir'] * 100:.1f}%")
cols[2].metric("Eligible loan (policy)", inr(m["eligible_loan_amount"]))
cols[3].metric("LTV", f"{m['ltv'] * 100:.1f}%" if m["ltv"] is not None else "—")

st.subheader("Risk flags")
for f in result["flags"]:
    color = LEVEL_COLOR[f["level"]]
    st.markdown(
        f"<div class='flag-card' style='border-left-color:{color}; background:{LEVEL_BG[f['level']]}'>"
        f"<span class='flag-title'>{LEVEL_ICON[f['level']]} {f['name']}"
        f"<span class='pill' style='background:{color}'>{f['level'].upper()}</span></span>"
        f"<div class='flag-reason'>{f['reason']}</div></div>",
        unsafe_allow_html=True,
    )

st.subheader("Underwriter narrative")
with st.spinner("Composing narrative…"):
    text, source = narrative.compose(result)
st.write(text)
if source == "fallback":
    st.caption("ℹ️ Deterministic fallback narrative (no LLM key). Set a provider key in `.env` for an LLM narrative.")

export = {**result, "narrative": text}
st.download_button(
    "⬇️ Download assessment (JSON)",
    data=json.dumps(export, indent=2),
    file_name="underwriting_assessment.json",
    mime="application/json",
)
