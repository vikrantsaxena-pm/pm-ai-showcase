"""Bank Statement Parser — Streamlit app.

Upload a PDF bank statement → extract transactions → auto-categorize each one →
see a cashflow summary (total in / total out / net) and a category breakdown.

Provider-agnostic: the model + API key come from the environment. See
providers.py and .env.example.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import providers

load_dotenv()

st.set_page_config(page_title="Bank Statement Parser", page_icon="🧾", layout="wide")

RUPEE = "₹"


def rupee(x: float) -> str:
    return f"{RUPEE}{x:,.2f}"


def read_pdf_text(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Provider:** `{providers.active_provider()}`")
    st.write(f"**Model:** `{providers.active_model() or 'not set'}`")
    st.caption(
        "Set `LLM_PROVIDER`, `LLM_MODEL`, and the matching API key in a `.env` "
        "file. Supported providers: Claude, OpenAI, Gemini. See `.env.example`."
    )

# --- Header ------------------------------------------------------------------
st.title("🧾 Bank Statement Parser")
st.write(
    "Upload a bank or UPI **PDF statement**. The app extracts every transaction, "
    "categorizes it (food, transport, bills, transfers, income, …), and shows your "
    "cashflow at a glance."
)

uploaded = st.file_uploader("Upload a PDF statement", type=["pdf"])

if uploaded is None:
    st.info("Upload a text-based PDF statement to begin. Scanned image PDFs are not supported.")
    st.stop()

with st.spinner("Reading PDF…"):
    try:
        text = read_pdf_text(uploaded)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the user
        st.error(f"Could not read the PDF: {exc}")
        st.stop()

if not text:
    st.error(
        "No selectable text found in this PDF. It's likely a scanned image — "
        "this app needs a text-based statement (OCR is out of scope)."
    )
    st.stop()

st.success(f"Extracted {len(text):,} characters of text from the statement.")
with st.expander("Preview extracted text"):
    st.text(text[:3000] + ("…" if len(text) > 3000 else ""))

if not st.button("Analyze transactions", type="primary"):
    st.stop()

with st.spinner(f"Categorizing with {providers.active_provider()}…"):
    try:
        txns = providers.extract_transactions(text)
    except providers.ProviderError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001 - LLM/network errors reach the user
        st.error(f"Extraction failed: {exc}")
        st.stop()

if not txns:
    st.warning("No transactions were found in this statement.")
    st.stop()

df = pd.DataFrame(txns)
df["signed_amount"] = df.apply(
    lambda r: r["amount"] if r["direction"] == "in" else -r["amount"], axis=1
)

summary = providers.cashflow_summary(txns)

# --- Cashflow summary --------------------------------------------------------
st.subheader("Cashflow summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total in", rupee(summary["total_in"]))
c2.metric("Total out", rupee(summary["total_out"]))
c3.metric("Net", rupee(summary["net"]), delta=rupee(summary["net"]))
c4.metric("Transactions", f"{summary['count']:,}")

# --- Spending by category ----------------------------------------------------
st.subheader("Spending by category")
spend = (
    df[df["direction"] == "out"]
    .groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)
if spend.empty:
    st.caption("No outgoing transactions to break down.")
else:
    st.bar_chart(spend)

# --- Transactions table ------------------------------------------------------
st.subheader("Transactions")
display = df[["date", "description", "category", "direction", "amount"]].copy()
display["amount"] = display["amount"].map(rupee)
st.dataframe(display, use_container_width=True, hide_index=True)

st.download_button(
    "Download as CSV",
    data=df[["date", "description", "category", "direction", "amount"]].to_csv(index=False),
    file_name="transactions.csv",
    mime="text/csv",
)
