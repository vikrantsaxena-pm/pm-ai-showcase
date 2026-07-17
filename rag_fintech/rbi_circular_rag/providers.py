"""Provider-agnostic embeddings and LLM generation, configured via env.

Embeddings (EMBED_PROVIDER):
  local   -> fastembed / BAAI/bge-small-en-v1.5  (runs on-device, no key)
  openai  -> text-embedding-3-small
  gemini  -> text-embedding-004

Generation (LLM_PROVIDER):
  claude  -> claude-opus-4-8   (Anthropic)
  openai  -> gpt-4o
  gemini  -> gemini-2.5-flash

Keys: ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or GOOGLE_API_KEY).
Heavy SDKs are imported lazily so the app launches without every provider installed.
"""

from __future__ import annotations

import os

import numpy as np

EMBED_DEFAULTS = {
    "local": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
}
LLM_DEFAULTS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}

_BATCH = 64  # batch size for hosted embedding APIs


class ProviderError(RuntimeError):
    """Configuration/API error with a user-friendly message."""


# --- Env helpers -------------------------------------------------------------
def embed_provider() -> str:
    return os.getenv("EMBED_PROVIDER", "local").strip().lower()


def embed_model() -> str:
    return os.getenv("EMBED_MODEL", "").strip() or EMBED_DEFAULTS.get(embed_provider(), "")


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "claude").strip().lower()


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "").strip() or LLM_DEFAULTS.get(llm_provider(), "")


def _require_key(*names: str) -> str:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    raise ProviderError(
        f"Missing API key. Set one of: {', '.join(names)} in your environment (.env)."
    )


# --- Embeddings --------------------------------------------------------------
_local_embedder = None


def _embed_local(texts: list[str]) -> np.ndarray:
    global _local_embedder
    from fastembed import TextEmbedding

    if _local_embedder is None:
        _local_embedder = TextEmbedding(model_name=embed_model())
    return np.asarray(list(_local_embedder.embed(texts)), dtype="float32")


def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        resp = client.embeddings.create(model=embed_model(), input=texts[i : i + _BATCH])
        out.extend(d.embedding for d in resp.data)
    return np.asarray(out, dtype="float32")


def _embed_gemini(texts: list[str]) -> np.ndarray:
    from google import genai

    client = genai.Client(api_key=_require_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        resp = client.models.embed_content(model=embed_model(), contents=texts[i : i + _BATCH])
        out.extend(e.values for e in resp.embeddings)
    return np.asarray(out, dtype="float32")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized float32 embeddings (so inner product == cosine)."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    provider = embed_provider()
    if provider == "local":
        vecs = _embed_local(texts)
    elif provider == "openai":
        vecs = _embed_openai(texts)
    elif provider in ("gemini", "google"):
        vecs = _embed_gemini(texts)
    else:
        raise ProviderError(
            f"Unknown EMBED_PROVIDER '{provider}'. Use one of: local, openai, gemini."
        )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype("float32")


# --- Generation --------------------------------------------------------------
def generate(system: str, user: str) -> str:
    provider = llm_provider()
    model = llm_model()
    if provider in ("claude", "anthropic"):
        return _gen_claude(model, system, user)
    if provider == "openai":
        return _gen_openai(model, system, user)
    if provider in ("gemini", "google"):
        return _gen_gemini(model, system, user)
    raise ProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: claude, openai, gemini."
    )


def _gen_claude(model: str, system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _gen_openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _gen_gemini(model: str, system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return (resp.text or "").strip()
