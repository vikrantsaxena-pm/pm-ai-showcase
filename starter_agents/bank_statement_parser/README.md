# 🧾 Bank Statement Parser

## Problem
People want to understand where their money goes, but bank statements are a wall
of cryptic narration lines — `UPI/DR/40481.../SWIGGY`, `NEFT CR SALARY`,
`ATM WDL`. Manually tagging each line to figure out monthly cashflow is tedious,
and generic budgeting apps often can't read an arbitrary bank's PDF layout.

## What it does
Upload a text-based PDF bank/UPI statement. The app:
1. Extracts the raw text from the PDF (`pypdf`).
2. Sends it to an LLM that pulls out every transaction and tags each one with a
   category tuned for Indian retail banking (Food & Dining, Groceries, Transport,
   Bills & Utilities, Transfers, Cash & ATM, Investments, Fees & Charges, Income, …).
3. Shows a **cashflow summary** (total in / total out / net), a **spending-by-category**
   breakdown, the full categorized transaction table, and a CSV export.

It is **provider-agnostic** — the model and API key are read from the environment,
and it works with **Claude, OpenAI, or Gemini** without code changes.

## Run steps
```bash
cd starter_agents/bank_statement_parser
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key

streamlit run app.py
```
Then open the URL Streamlit prints (default http://localhost:8501), upload a PDF,
and click **Analyze transactions**.

## Screenshot
<!-- Add a screenshot of the cashflow summary + category chart here -->
_Screenshot slot — run the app and drop an image here._

## PM notes
- **Metric moved:** time-to-insight on "where did my money go" — from a manual
  30-minute tagging exercise to a single upload and one click.
- **User job:** an individual reconciling monthly spend, or a lending/underwriting
  analyst who needs a quick categorized read of an applicant's statement.
- **Trade-off made:** we rely on the LLM to parse arbitrary bank layouts (robust to
  format variety) instead of hand-written per-bank regex — this trades a small
  per-run API cost and occasional mis-categorization for working on statements we've
  never seen. Scanned/image PDFs are explicitly out of scope (no OCR); the app
  detects them and tells the user rather than silently returning nothing.
