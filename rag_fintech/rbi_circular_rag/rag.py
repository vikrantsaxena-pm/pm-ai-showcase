"""RAG core: PDF -> chunks -> FAISS vector store -> grounded, cited answers.

The guardrail lives here: retrieval is the ONLY source of truth. If the
retrieved excerpts don't contain the answer, the model is instructed to reply
with the exact GUARDRAIL sentence instead of using its own knowledge, and a
relevance floor can short-circuit before the model is ever called.
"""

from __future__ import annotations

import json
import os
import re

import faiss
import numpy as np
from pypdf import PdfReader

import providers

STORE_DIR = os.getenv("RAG_STORE_DIR", ".vectorstore")
GUARDRAIL = "This isn't covered in the loaded documents."

SYSTEM_PROMPT = (
    "You are a compliance research assistant. Answer the user's question using ONLY "
    "the numbered context excerpts below, which are quotations from official RBI "
    "documents the user has loaded.\n\n"
    "Strict rules:\n"
    "1. Use only information present in the excerpts. Do NOT use outside or prior "
    "knowledge about RBI, banking, or finance.\n"
    f'2. If the excerpts do not clearly contain the answer, reply with EXACTLY this '
    f'sentence and nothing else: "{GUARDRAIL}"\n'
    "3. Do not guess, infer beyond the text, or fill gaps from general knowledge.\n"
    "4. When you answer, cite the excerpt numbers you used in square brackets, e.g. [2].\n"
    "5. Quote figures, dates, limits, and thresholds exactly as written."
)


# --- PDF -> chunks -----------------------------------------------------------
def _extract_pages(file, name: str) -> list[tuple[int, str]]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring sentence/line boundaries."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            brk = max(window.rfind(". "), window.rfind("\n"))
            if brk > size * 0.5:
                end = start + brk + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(file, name: str, size: int = 1000, overlap: int = 150) -> list[dict]:
    chunks = []
    for page_no, text in _extract_pages(file, name):
        for piece in chunk_text(text, size, overlap):
            chunks.append({"source": name, "page": page_no, "text": piece})
    return chunks


# --- Vector store (FAISS, cosine via inner product on normalized vectors) -----
class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self.dim: int | None = None
        self.provider: str | None = None
        self.model: str | None = None

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        cur_provider, cur_model = providers.embed_provider(), providers.embed_model()
        if self.provider and (self.provider, self.model) != (cur_provider, cur_model):
            raise providers.ProviderError(
                f"Index was built with embeddings '{self.provider}/{self.model}' but the "
                f"current config is '{cur_provider}/{cur_model}'. Clear the index to rebuild."
            )
        vecs = providers.embed_texts([c["text"] for c in chunks])
        if self.index is None:
            self.dim = int(vecs.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
            self.provider, self.model = cur_provider, cur_model
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        q = providers.embed_texts([query])
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            hits.append(hit)
        return hits

    # persistence ------------------------------------------------------------
    def save(self, path: str = STORE_DIR) -> None:
        if self.index is None:
            return
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(
                {"chunks": self.chunks, "dim": self.dim,
                 "provider": self.provider, "model": self.model},
                fh,
            )

    @classmethod
    def load(cls, path: str = STORE_DIR) -> "VectorStore | None":
        idx_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return None
        with open(meta_path) as fh:
            meta = json.load(fh)
        store = cls()
        store.index = faiss.read_index(idx_path)
        store.chunks = meta["chunks"]
        store.dim = meta["dim"]
        store.provider = meta["provider"]
        store.model = meta["model"]
        return store

    @staticmethod
    def clear(path: str = STORE_DIR) -> None:
        for name in ("index.faiss", "meta.json"):
            fp = os.path.join(path, name)
            if os.path.exists(fp):
                os.remove(fp)


# --- Ask ---------------------------------------------------------------------
def answer(store: VectorStore, question: str, k: int = 5, min_score: float = 0.0) -> dict:
    """Retrieve, then answer strictly from the retrieved excerpts.

    Returns {answer, grounded, citations}. `grounded` is False when the model
    (or the relevance floor) fell back to the guardrail message.
    """
    hits = store.search(question, k=k)
    if not hits or (min_score > 0 and hits[0]["score"] < min_score):
        return {"answer": GUARDRAIL, "grounded": False, "citations": hits}

    context = "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    user = f"Context excerpts:\n{context}\n\nQuestion: {question}"
    text = providers.generate(SYSTEM_PROMPT, user)
    grounded = GUARDRAIL.lower() not in text.lower()
    return {"answer": text, "grounded": grounded, "citations": hits}
