# 🏦 Credit Underwriting Copilot

## Problem
Underwriters make the same calculations on every file — EMI, FOIR/DTI, eligible
loan amount, LTV — then translate those numbers into a risk call and a written
rationale. Doing it by hand is slow and inconsistent, but you also **cannot** let
an LLM do the arithmetic or the decision: a hallucinated FOIR or a made-up
approval is a compliance incident. The math must be exact and auditable.

## What it does
An underwriter enters (or uploads) applicant data — income, existing
obligations, bureau score, employment type, loan amount, tenure, rate, and
optional asset value. Then:

1. **Income verification (optional).** The underwriter uploads a bank statement
   PDF (or pastes statement text). The **LLM extracts structured income signals**
   from that unstructured text — salary-like recurring credits (amount, date,
   payer), irregular/cash-like inflows, and the detected employer. Then
   **deterministic Python** (`income.py`) reconciles: it computes observed
   monthly income (median of monthly salary credits), income stability (how many
   of the last N months have a salary credit + the coefficient of variation of
   the amounts), and the declared-vs-observed **variance flag**. The extraction
   is shown **side-by-side with the declared figures**, so it's auditable, not
   trusted blindly.
2. **Choose the income basis** — assess on *declared*, *observed*, or the *lower
   of the two*. FOIR and eligibility are then computed on that verified income.
3. **Deterministic Python tools** (`underwriting.py`) compute EMI (reducing
   balance), FOIR/DTI, policy-eligible loan amount, max affordable EMI, and LTV,
   and assign a **red / amber / green flag with a reason** to each dimension
   (repayment capacity, eligibility, bureau, income variance, income stability,
   employment, LTV). The overall recommendation is the worst flag.
4. **The LLM writes the narrative** from the computed JSON — forbidden from
   changing any figure or the decision. No key? A deterministic fallback
   narrative is used, so the app is fully functional offline.

Provider-agnostic: `LLM_PROVIDER` selects Claude / OpenAI / Gemini via env.

### Architecture (why the split matters)
```
                    unstructured statement text
                              │
                    LLM: extract income signals   ◀── job (a): read text only
                              │  (salary credits, inflows, employer)
                              ▼
applicant ─▶ income.py ─▶ underwriting.py ─▶ metrics + flags + recommendation ─▶ narrative.py ─▶ LLM
             (observed income,   (all loan math & rules,   (deterministic,          (prose only,   ── job (b)
              variance, stability,  folds in income flags)   auditable)              no numbers)
              all pure Python)
```
The LLM does exactly two things: **(a)** turn statement text into structured
signals, and **(b)** write prose from the final JSON. Every number, every flag,
and the approve/refer/decline call are produced by deterministic code you can
unit-test — which is exactly what `test_underwriting.py` does.

## Run steps
```bash
cd advanced_agents/credit_underwriting_copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional: set LLM_PROVIDER + a key for LLM narrative
streamlit run app.py
```
In the app: click **Load sample applicant**, then in the income section
**Load sample statement text** and either **Extract income signals** (needs an
LLM key) or **Use sample extraction (offline)** (no key — loads the bundled
ground-truth extraction). Pick an income basis and **Assess**.

### Fixtures (all clearly fictional)
- `sample_applicant.json` — declared applicant (declares ₹1,00,000/month).
- `sample_statement.txt` — a fictional 6-month statement whose **true income is
  ₹85,000/month** (salary from "ACME TECHNOLOGIES PVT LTD"), plus a cash deposit
  and a one-off UPI receipt that must be classified as *irregular*, not salary.
- `sample_statement_signals.json` — the ground-truth extraction of that statement
  (used for the offline demo button and by the tests).

With the sample applicant + sample statement, observed (₹85,000) is 15% below
declared (₹1,00,000) → an **amber** income-variance flag, while the identical
monthly amounts give **green** stability.

### Run the offline tests (no key, no network, no LLM)
```bash
python test_underwriting.py
```
18 tests. Loan math: EMI (₹22,244.45 for ₹10L @12%/60m), FOIR (32.24%), eligible
(≈₹17.98L), LTV (80%), EMI↔present-value inverse. Income layer: observed income
from the fixture (₹85,000), stability coverage + coefficient of variation
(hand-verified CoV 0.0326599 for 48k/50k/52k), the variance flag at its exact
amber/red **threshold boundaries**, income basis selection, and that a red income
flag propagates into the overall recommendation. No test calls the LLM.

## Screenshot
<!-- Add a screenshot of the recommendation banner + flags + metrics here -->
_Screenshot slot — run the app and drop an image here._

## PM notes
- **Metric moved:** underwriting cycle time and income-fraud catch rate — the
  copilot verifies declared income against the actual bank statement and flags
  overstatement automatically, instead of an analyst eyeballing salary credits.
- **User job:** a credit underwriter deciding approve / refer / decline, who must
  verify stated income and produce a defensible written rationale.
- **Trade-off made:** we deliberately keep **all math and all decisions in
  deterministic Python** and use the LLM only for the narrative. This costs some
  flexibility (the model can't reason its way to a different number) but buys
  auditability and eliminates the worst failure mode — a confident, wrong
  financial figure. Policy thresholds (max FOIR, score bands, LTV cap) are
  **lender-configurable, not RBI mandates**; RBI does publish tiered LTV caps for
  some secured loans (e.g. home loans ~90/80/75% by ticket size, gold ~75%) —
  set the cap in the sidebar to match current RBI norms for your product rather
  than trusting a hard-coded value.
