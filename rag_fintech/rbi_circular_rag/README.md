# 📑 RBI Circular RAG

## Problem
RBI Master Directions and circulars (e-mandate / recurring-payments framework,
KYC, PA/PG guidelines) are long, cross-referencing PDFs. Teams need to answer
precise compliance questions — *"what's the AFA limit for e-mandates?"* — from
the **actual text of the regulation**, not a model's fuzzy recollection. A wrong
but confident answer about a regulation is worse than no answer.

## What it does
- Load one or more RBI PDFs (Master Direction, e-mandate framework, any circular).
- Chunks each PDF (per page, with overlap) and embeds the chunks into a **local
  FAISS vector store** — no hosted database required.
- Answers your questions **only from the retrieved chunks**, and shows the
  **source citations** for every answer (document name + page + the exact chunk text).
- **Guardrail:** if the retrieved chunks don't contain the answer, it replies
  *"This isn't covered in the loaded documents."* instead of answering from the
  model's general knowledge. This is enforced by a strict grounding prompt (and an
  optional relevance floor, `RAG_MIN_SCORE`).

Provider-agnostic: the **LLM** (Claude / OpenAI / Gemini) and **embeddings**
(local / OpenAI / Gemini) are chosen via env. The default embeddings run
**on-device** (fastembed), so retrieval works without any embedding API key —
you only need a key for the answer-generation LLM.

## Run steps
```bash
cd rag_fintech/rbi_circular_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and its API key. Leave EMBED_PROVIDER=local to
# run embeddings on-device (first run downloads a small model).

streamlit run app.py
```
Then in the app: **upload a PDF → "Add to knowledge base" → ask a question.**

### Getting a sample RBI PDF
The app's **uploader is the primary way to load documents** — download a circular
in your browser and upload it. RBI's site is often behind a bot/CAPTCHA check, so
scripted downloads can fail.

- Browse **https://www.rbi.org.in** → *Notifications* or *Master Directions*.
- Example — e-mandate / processing of recurring transactions notification:
  **https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12051**
- Helper for direct PDF links (verifies it's really a PDF, else tells you to
  download manually): `python fetch_sample.py <pdf-url>` → saves into `data/`.

> Only text-based PDFs are supported (no OCR). Scanned image PDFs are detected and skipped.

## Screenshot
<!-- Add a screenshot of an answer with its citations panel here -->
_Screenshot slot — run the app and drop an image here._

## PM notes
- **Metric moved:** answer *trustworthiness* on regulatory questions — every
  answer is traceable to a specific page/chunk, and unanswerable questions are
  refused rather than hallucinated.
- **User job:** a compliance/payments analyst checking what a specific RBI
  circular actually mandates, without reading the whole PDF.
- **Trade-off made:** we hard-limit answers to retrieved text (strict grounding +
  refusal). This deliberately trades *coverage* (it won't answer from world
  knowledge, and won't stitch across documents it didn't retrieve) for
  *faithfulness* — the right trade-off when a confident-but-wrong regulatory
  answer is the expensive failure. `RAG_MIN_SCORE` lets you make refusal stricter.
