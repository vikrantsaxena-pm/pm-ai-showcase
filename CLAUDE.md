# House style for this repo

## What this repo is
Runnable AI agent / RAG / MCP apps focused on payments, fintech, and Indian
govtech. Every app must run with: clone → install → run.

## Conventions (apply to every app)
- Front-end: Streamlit (`app.py` is the entry point).
- Provider-agnostic: model + API key read from env; support at least two of
  {Claude, Gemini, OpenAI}. Never hardcode a provider.
- Each app folder MUST contain: app.py, requirements.txt (pinned),
  .env.example, README.md.
- requirements.txt must be tested — run the app before considering it done.
- README.md structure: Problem, What it does, Run steps, Screenshot slot,
  PM notes (2-3 lines: metric moved, user job, trade-off made).
- Keep secrets out of git. .env.example documents keys with placeholder values.

## Do
- Make the domain logic actually correct (real RBI rules, real lending flow).
- Keep each app self-contained.

## Don't
- Don't invent fintech facts. If a regulation detail is uncertain, flag it.
- Don't ship an app you haven't run successfully.

## Exceptions
- `visa_document_validator/` is a full product (FastAPI + Supabase + crawler +
  two Streamlit clients), NOT a self-contained Streamlit demo. Inside that
  folder, its own `BUILD_SPEC.md` is authoritative and supersedes the
  conventions above (e.g. no `app.py` Streamlit entry point).
