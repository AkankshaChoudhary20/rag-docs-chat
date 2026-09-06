"""Query-time retrieval: embed the question, search pgvector for top-k chunks."""

from __future__ import annotations

from typing import Any

from . import db
from .config import get_settings
from .embeddings import embed_query


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    k = top_k or settings.top_k
    query_vector = embed_query(question)
    return db.search_similar_chunks(query_vector, top_k=k)
