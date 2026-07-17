"""RBI Circular RAG — ask questions answered only from loaded RBI documents.

Load one or more RBI PDFs (Master Direction, e-mandate / recurring-payments
framework, etc.). The app chunks + embeds them into a local FAISS store and
answers questions strictly from the retrieved excerpts, with citations. If the
excerpts don't contain the answer, it says so instead of guessing.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import providers
import rag

load_dotenv()

st.set_page_config(page_title="RBI Circular RAG", page_icon="📑", layout="wide")

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))  # 0 = rely on the LLM guardrail


def get_store() -> rag.VectorStore:
    """Load a persisted store if it matches the current embedding config."""
    if "store" not in st.session_state:
        loaded = rag.VectorStore.load()
        cur = (providers.embed_provider(), providers.embed_model())
        if loaded is not None and (loaded.provider, loaded.model) == cur:
            st.session_state.store = loaded
        else:
            if loaded is not None:
                st.session_state.store_note = (
                    f"Ignored a saved index built with '{loaded.provider}/{loaded.model}' "
                    f"— current embeddings are '{cur[0]}/{cur[1]}'. Re-add your documents."
                )
            st.session_state.store = rag.VectorStore()
    return st.session_state.store


store = get_store()

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.write(f"**LLM:** `{providers.llm_provider()}` / `{providers.llm_model() or 'not set'}`")
    st.write(
        f"**Embeddings:** `{providers.embed_provider()}` / "
        f"`{providers.embed_model() or 'not set'}`"
    )
    st.write(f"**Top-K:** {TOP_K}  |  **Min score:** {MIN_SCORE}")
    st.caption(
        "Set `LLM_PROVIDER`, `EMBED_PROVIDER`, and matching keys in `.env`. "
        "Default embeddings run locally (no key needed); the LLM needs a key. "
        "See `.env.example`."
    )
    if store.count:
        st.divider()
        st.write(f"**Indexed:** {store.count} chunks")
        for s in store.sources:
            st.write(f"• {s}")
        if st.button("Clear index"):
            rag.VectorStore.clear()
            st.session_state.store = rag.VectorStore()
            st.rerun()

# --- Header ------------------------------------------------------------------
st.title("📑 RBI Circular RAG")
st.write(
    "Load RBI PDF documents, then ask questions answered **only** from what those "
    "documents actually say — with source citations. If the answer isn't in the "
    "loaded documents, the app tells you instead of guessing."
)

if note := st.session_state.pop("store_note", None):
    st.info(note)

# --- Load documents ----------------------------------------------------------
st.subheader("1. Load RBI documents")
uploads = st.file_uploader(
    "Upload one or more RBI PDFs (Master Direction, e-mandate framework, …)",
    type=["pdf"],
    accept_multiple_files=True,
)
st.caption(
    "No file handy? Download a circular from rbi.org.in (Notifications / Master "
    "Directions) and upload it, or use `python fetch_sample.py <pdf-url>`. See the README."
)

if uploads and st.button("Add to knowledge base", type="primary"):
    added = 0
    with st.spinner("Chunking and embedding…"):
        for up in uploads:
            try:
                chunks = rag.build_chunks(up, up.name)
                if not chunks:
                    st.warning(f"No selectable text in **{up.name}** (scanned image?). Skipped.")
                    continue
                store.add(chunks)
                added += len(chunks)
            except providers.ProviderError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to process {up.name}: {exc}")
                st.stop()
    if added:
        store.save()
        st.success(f"Added {added} chunks. Knowledge base now has {store.count} chunks.")

# --- Ask ---------------------------------------------------------------------
st.subheader("2. Ask a question")
if not store.count:
    st.info("Load at least one document above to start asking questions.")
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the additional factor of authentication limit for e-mandates?",
)

if question and st.button("Ask", type="primary"):
    with st.spinner(f"Searching documents and answering with {providers.llm_provider()}…"):
        try:
            result = rag.answer(store, question, k=TOP_K, min_score=MIN_SCORE)
        except providers.ProviderError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to answer: {exc}")
            st.stop()

    st.markdown("### Answer")
    if result["grounded"]:
        st.markdown(result["answer"])
    else:
        st.warning(result["answer"])

    citations = result["citations"]
    if citations:
        label = (
            "Sources" if result["grounded"] else "Closest excerpts (did not contain the answer)"
        )
        st.markdown(f"### {label}")
        for i, c in enumerate(citations, start=1):
            with st.expander(f"[{i}] {c['source']} — page {c['page']}  (score {c['score']:.3f})"):
                st.write(c["text"])
