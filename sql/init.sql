-- rag-docs-chat: schema initialization
-- Runs automatically on first container start via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    filename      TEXT NOT NULL,
    content_type  TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    num_chunks    INTEGER NOT NULL DEFAULT 0
);

-- 384 dimensions matches sentence-transformers/all-MiniLM-L6-v2.
-- If you swap embedding models, update this dimension to match.
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- HNSW index for fast approximate cosine-similarity search. Unlike IVFFlat,
-- HNSW doesn't need pre-existing data to build a useful index, which suits
-- a demo app where the table starts empty.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
