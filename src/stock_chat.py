"""Stock-module chat using Groq (free tier).

Talks to Groq's OpenAI-compatible Chat Completions API directly via the `groq`
SDK (pulled in transitively by `langchain-groq`). Kept independent of the
LangChain RAG pipeline used by the medical tab so the two flows don't share
a fragile dependency chain.

Production additions vs. v1:
  - _make_stream(): retries stream creation up to 2× with exponential back-off
    (covers transient 429 / 503 errors; skips retry on auth / bad-request).
  - call_groq_json(): non-streaming call for structured JSON responses
    (used by the Resume Analyzer to get reliable JSON back).
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Iterable, Iterator

from groq import Groq

MAX_TOKENS = 1500
MAX_TOKENS_JSON = 2500   # structured analysis needs more headroom


# Reuse the same Groq models the medical tab supports.
# Llama 3.3 70B is the strongest free option for analysis-style answers.
STOCK_LLM_MODELS = {
    "GPT-OSS 120B (strongest reasoning)": "openai/gpt-oss-120b",
    "Llama 3.3 70B (balanced)":            "llama-3.3-70b-versatile",
    "Qwen 3 32B (reasoning)":              "qwen/qwen3-32b",
    "Llama 3.1 8B (fastest)":              "llama-3.1-8b-instant",
}

DEFAULT_MODEL = next(iter(STOCK_LLM_MODELS.values()))

# Errors that should NOT be retried (client-side mistakes).
_NO_RETRY_SIGNALS = ("401", "403", "400", "invalid api key", "bad request", "model_decommissioned")


def _client(api_key: str | None = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
    return Groq(api_key=key)


def build_system_prompt(template: str, psx_data: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return template.format(psx_data=psx_data, today=today)


# ---------------------------------------------------------------------------
# Internal retry helper
# ---------------------------------------------------------------------------
def _make_stream(client: Groq, model: str, messages: list[dict], temperature: float):
    """Create a Groq streaming completion, retrying up to 2× on transient errors."""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
        except Exception as exc:
            last_err = exc
            err_lower = str(exc).lower()
            if any(sig in err_lower for sig in _NO_RETRY_SIGNALS):
                raise  # don't retry auth / bad-request errors
            if attempt < 2:
                time.sleep(2 ** attempt)   # 1s, then 2s
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------
def stream_chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> Iterator[str]:
    """Yield response text chunks from Groq.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts.
    The new user_message is appended; the system prompt is sent separately.
    Retries stream creation up to 2× on transient server errors.
    """
    messages = (
        [{"role": "system", "content": system_prompt}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )

    client = _client(api_key)
    stream = _make_stream(client, model, messages, temperature)

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Non-streaming JSON call (for structured outputs like resume analysis)
# ---------------------------------------------------------------------------
def call_groq_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    """Non-streaming Groq call intended for structured JSON responses.

    Uses temperature=0.1 for reliability. Retries up to 2× on transient errors.
    Returns the raw response string (the caller parses JSON).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    client = _client(api_key)
    last_err: Exception | None = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_TOKENS_JSON,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            err_lower = str(exc).lower()
            if any(sig in err_lower for sig in _NO_RETRY_SIGNALS):
                raise
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------
def history_for_chat(messages: Iterable[dict], max_turns: int = 4) -> list[dict]:
    """Filter Streamlit-style messages into the role/content shape Groq expects.

    Keeps only the last `max_turns * 2` user/assistant messages and drops any
    stub roles or blanks.
    """
    recent = list(messages)[-(max_turns * 2):]
    out: list[dict] = []
    for m in recent:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out
