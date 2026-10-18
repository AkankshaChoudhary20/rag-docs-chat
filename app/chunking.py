"""Simple sliding-window text chunker.

Chunks are measured in characters (not model tokens) to keep the
implementation dependency-free and predictable. ~500 characters is a
reasonable proxy for a few hundred tokens of English prose, which is the
target chunk size for retrieval.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split `text` into overlapping chunks using a sliding window.

    Args:
        text: The raw text to split.
        chunk_size: Target number of characters per chunk.
        overlap: Number of characters each chunk shares with the previous one.

    Returns:
        A list of non-empty, whitespace-normalized chunks. Returns an empty
        list if `text` is empty/whitespace-only.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if type(overlap) is not int or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    # Normalize whitespace so page breaks / repeated newlines from PDF
    # extraction don't blow up chunk boundaries.
    normalized = " ".join(text.split())
    if not normalized:
        return []

    if overlap >= chunk_size:
        # Guard against a degenerate config that would never advance.
        overlap = chunk_size // 4

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks
