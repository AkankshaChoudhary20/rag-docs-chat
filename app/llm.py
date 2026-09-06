"""Calls the Anthropic Claude API with retrieved chunks as grounding context."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import anthropic

from .config import get_settings

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the \
provided document excerpts (the "context"). Each excerpt is labeled with a citation tag \
like [1], [2], etc.

Rules:
- Base your answer only on the provided context. If the context does not contain enough \
information to answer, say so plainly instead of guessing.
- Cite the excerpt(s) you used inline using their tags, e.g. "The API key is read from an \
environment variable [2]."
- Be concise and direct.
"""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file "
            "(see .env.example) to enable /chat."
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f'[{i}] (source: {chunk["filename"]}, chunk #{chunk["chunk_index"]})\n'
            f'{chunk["content"]}'
        )
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[dict[str, Any]]) -> str:
    """Ask Claude to answer `question` using `chunks` as the only context.

    Returns the model's text answer. Raises `RuntimeError` if no API key is
    configured, and re-raises any `anthropic.APIError` subclass on failure.
    """
    settings = get_settings()
    client = get_client()

    if not chunks:
        context_block = "(No relevant document excerpts were found.)"
    else:
        context_block = build_context_block(chunks)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above, and cite the excerpt "
        "numbers (e.g. [1]) you relied on."
    )

    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("Could not reach the Anthropic API. Check your network.") from exc

    if response.stop_reason == "refusal":
        return "The model declined to answer this question."

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts).strip() or "(The model returned an empty response.)"
