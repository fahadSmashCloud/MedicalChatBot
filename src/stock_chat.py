"""Stock-module chat using Groq (free tier).

Talks to Groq's OpenAI-compatible Chat Completions API directly via the `groq`
SDK (pulled in transitively by `langchain-groq`). Kept independent of the
LangChain RAG pipeline used by the medical tab so the two flows don't share
a fragile dependency chain.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Iterator

from groq import Groq

MAX_TOKENS = 1500


# Reuse the same Groq models the medical tab supports.
# Llama 3.3 70B is the strongest free option for analysis-style answers.
STOCK_LLM_MODELS = {
    "Llama 3.3 70B (versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (instant)":    "llama-3.1-8b-instant",
    "Gemma 2 9B":                "gemma2-9b-it",
    "Llama 3 70B (legacy)":      "llama3-70b-8192",
}

DEFAULT_MODEL = next(iter(STOCK_LLM_MODELS.values()))


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
    """
    messages = (
        [{"role": "system", "content": system_prompt}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )

    client = _client(api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


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
