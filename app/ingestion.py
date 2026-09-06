"""End-to-end document ingestion: extract -> chunk -> embed -> store."""

from __future__ import annotations

from . import db
from .chunking import chunk_text
from .config import get_settings
from .embeddings import embed_texts
from .extraction import extract_text


class EmptyDocumentError(ValueError):
    """Raised when a document contains no extractable text."""


def ingest_document(filename: str, content_type: str | None, data: bytes) -> dict:
    """Run the full ingestion pipeline for one uploaded file.

    Returns a summary dict with the new document's id, filename, and the
    number of chunks stored.
    """
    settings = get_settings()

    text = extract_text(filename, content_type, data)
    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

    if not chunks:
        raise EmptyDocumentError(
            f"No extractable text found in '{filename}'. "
            "For PDFs, this can happen with scanned/image-only pages."
        )

    embeddings = embed_texts(chunks)

    document_id = db.insert_document(filename, content_type)
    db.insert_chunks(document_id, chunks, embeddings)
    db.set_document_chunk_count(document_id, len(chunks))

    return {
        "id": document_id,
        "filename": filename,
        "num_chunks": len(chunks),
    }
