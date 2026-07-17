"""Provider-agnostic LLM text generation, configured via env.

  LLM_PROVIDER   claude | openai | gemini   (default: claude)
  LLM_MODEL      overrides the per-provider default
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or GOOGLE_API_KEY)

The LLM is used ONLY to write the narrative from already-computed numbers.
All math and decisioning happen in underwriting.py.
"""

from __future__ import annotations

import json
import os

LLM_DEFAULTS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}


class ProviderError(RuntimeError):
    """Configuration/API error with a user-friendly message."""


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "claude").strip().lower()


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "").strip() or LLM_DEFAULTS.get(llm_provider(), "")


def has_api_key() -> bool:
    provider = llm_provider()
    keys = {
        "claude": ("ANTHROPIC_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(provider, ())
    return any(os.getenv(k) for k in keys)


def _require_key(*names: str) -> str:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    raise ProviderError(
        f"Missing API key. Set one of: {', '.join(names)} in your environment (.env)."
    )


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
        max_tokens=1500,
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


def extract_json(system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict:
    """Provider-agnostic structured extraction -> parsed dict matching `schema`.

    This is an LLM 'read unstructured text' job (not a math job): the model
    extracts fields; Python decides what to do with them.
    """
    provider = llm_provider()
    model = llm_model()
    if provider in ("claude", "anthropic"):
        import anthropic

        client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return json.loads(text)
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": True},
            },
        )
        return json.loads(resp.choices[0].message.content)
    if provider in ("gemini", "google"):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_require_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(resp.text)
    raise ProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: claude, openai, gemini."
    )
