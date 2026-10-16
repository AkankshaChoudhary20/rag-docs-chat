"""Postgres access layer (psycopg3 + pgvector).

A single module-level connection pool is created lazily on first use so
that importing this module (e.g. for tests) doesn't require a live
database connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from .config import get_settings

_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    """Register the pgvector type adapter on every new pool connection."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure_connection,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def check_connection() -> bool:
    """Used by /health -- returns True if the DB is reachable."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def insert_document(filename: str, content_type: str | None) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (filename, content_type)
            VALUES (%s, %s)
            RETURNING id
            """,
            (filename, content_type),
        ).fetchone()
        conn.commit()
        return row[0]


def store_document(filename: str, content_type: str | None, chunks: list[str],
                   embeddings: list[list[float]]) -> int:
    """Persist document metadata and all chunks in one transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")
    with get_conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO documents (filename, content_type, num_chunks) VALUES (%s, %s, %s) RETURNING id",
                (filename, content_type, len(chunks)),
            ).fetchone()
            document_id = row[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(document_id, index, chunk, embedding)
                     for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))],
                )
        return document_id


def set_document_chunk_count(document_id: int, num_chunks: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET num_chunks = %s WHERE id = %s",
            (num_chunks, document_id),
        )
        conn.commit()


def list_documents() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, content_type, uploaded_at, num_chunks
            FROM documents
            ORDER BY uploaded_at DESC
            """
        ).fetchall()
        columns = ["id", "filename", "content_type", "uploaded_at", "num_chunks"]
        return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def insert_chunks(
    document_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (document_id, idx, chunk, embedding)
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ],
            )
        conn.commit()


def search_similar_chunks(
    query_embedding: list[float], top_k: int = 4
) -> list[dict[str, Any]]:
    """Cosine-similarity nearest-neighbor search over stored chunks.

    pgvector's `<=>` operator returns *cosine distance* (0 = identical,
    2 = opposite). We convert to a similarity score (1 - distance) for
    display purposes.

    Note: the explicit `::vector` cast on the parameter is required. Without
    it, Postgres resolves the `<=>` operator against the parameter's default
    inferred type (a plain float array), which has no matching operator --
    pgvector only provides an *assignment* cast from float arrays to
    `vector` (usable on INSERT/UPDATE), not one usable in operator
    resolution.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.document_id,
                d.filename,
                c.chunk_index,
                c.content,
                1 - (c.embedding <=> %(query)s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %(query)s::vector
            LIMIT %(top_k)s
            """,
            {"query": query_embedding, "top_k": top_k},
        ).fetchall()
        columns = ["id", "document_id", "filename", "chunk_index", "content", "similarity"]
        return [dict(zip(columns, row)) for row in rows]
