# rag-docs-chat

A small, self-contained Retrieval-Augmented Generation (RAG) app: upload a document,
ask questions about it, and get answers grounded in your own text with cited sources.

No hosted vector database, no external embeddings API, no build step for the frontend.
Embeddings run **locally** on CPU via `sentence-transformers`; only the final answer
generation calls out to the Claude API.

## What it does

1. **Ingest** — upload a `.txt`, `.md`, or `.pdf` file. The backend extracts the raw
   text (via `pypdf` for PDFs), splits it into overlapping ~500-character chunks, embeds
   each chunk locally with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions),
   and stores the chunks + embeddings in Postgres using the `pgvector` extension.
2. **Ask** — submit a question. The backend embeds the question with the same local
   model, runs a cosine-similarity nearest-neighbor search in pgvector to find the most
   relevant chunks, and sends those chunks as context to Claude, asking it to answer
   using only that context and to cite which excerpt(s) it used.
3. **Answer** — the response includes Claude's answer plus the retrieved source chunks
   (filename, chunk index, similarity score, and the chunk text itself) so you can verify
   the grounding yourself.

A minimal HTML/JS frontend (no React, no bundler — just `fetch()` calls) is served
directly by FastAPI at `/`, so the whole thing is usable in a browser with zero frontend
tooling.

## Architecture

```
                    ┌──────────────────────────┐
  Browser  ───────► │  FastAPI (app/main.py)   │
  (static           │                          │
   HTML/JS at "/")  │  POST /documents ────────┼──► extraction.py (pypdf / plain text)
                     │                          │         │
                     │                          │         ▼
                     │                          │    chunking.py (sliding window, ~500 chars)
                     │                          │         │
                     │                          │         ▼
                     │                          │    embeddings.py (sentence-transformers,
                     │                          │         │          all-MiniLM-L6-v2, local CPU)
                     │                          │         ▼
                     │                          │    db.py ──► Postgres + pgvector
                     │                          │              (documents, chunks tables)
                     │                          │
                     │  POST /chat ─────────────┼──► retrieval.py (embed question,
                     │                          │         │        cosine <=> search in pgvector)
                     │                          │         ▼
                     │                          │    llm.py ──► Anthropic Claude API
                     │                          │              (answer + citations, using
                     │                          │               retrieved chunks as context)
                     │                          │
                     │  GET /documents          │
                     │  GET /health             │
                     └──────────────────────────┘
```

Two data stores of note, both in the `chunks` table:

- `content` (text) — the raw chunk text, used as the LLM context and shown in the UI.
- `embedding` (`vector(384)`) — the local embedding, indexed with an HNSW pgvector index
  for fast approximate cosine-similarity search (`embedding <=> query_vector`).

## Stack

- **Backend**: FastAPI + Uvicorn
- **Vector storage**: PostgreSQL 16 + [`pgvector`](https://github.com/pgvector/pgvector)
  (via the `pgvector/pgvector:pg16` Docker image)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, runs locally —
  no API key needed for this part)
- **PDF parsing**: `pypdf`
- **LLM**: Anthropic Claude API (`claude-sonnet-5` by default, configurable) via the
  official `anthropic` Python SDK
- **Frontend**: a single static HTML file with vanilla JS `fetch()` calls, served by
  FastAPI — no build step

## Setup

### Prerequisites

- Python 3.11+ (the project uses `sentence-transformers`/`torch`, which need a Python
  version those packages currently ship wheels for — 3.11 or 3.12 is the safest bet)
- Docker + Docker Compose (for Postgres with pgvector)
- An Anthropic API key — get one at
  [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

### 1. Start Postgres with pgvector

```bash
docker compose up -d
```

This starts a `pgvector/pgvector:pg16` container on `localhost:5432` and runs
`sql/init.sql` on first boot, which creates the `vector` extension and the `documents`
and `chunks` tables (with an HNSW index for cosine similarity search).

### 2. Set up a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first `sentence-transformers` import will download the `all-MiniLM-L6-v2` model
(~90MB) from Hugging Face and cache it locally — this happens once.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` to your key. The default `DATABASE_URL`
already matches the `docker-compose.yml` defaults, so you shouldn't need to change it
unless you changed the compose file.

### 4. Run the app

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. Upload a `.txt`/`.md`/`.pdf` file, wait
for it to be ingested, then ask a question about it.

Interactive API docs (Swagger UI) are available at **http://localhost:8000/docs**.

## API reference

| Method | Path         | Description                                      |
|--------|--------------|---------------------------------------------------|
| GET    | `/`          | Serves the static frontend                         |
| GET    | `/health`    | Health check (DB connectivity, config status)       |
| POST   | `/documents` | Upload + ingest a `.txt`/`.md`/`.pdf` file          |
| GET    | `/documents` | List ingested documents                             |
| POST   | `/chat`      | Ask a question, get an answer + cited source chunks |

### Example curl commands

Check health:

```bash
curl http://localhost:8000/health
```

Upload a document:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@/path/to/your/document.pdf"
```

List ingested documents:

```bash
curl http://localhost:8000/documents
```

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Example response shape:

```json
{
  "answer": "The document describes... [1]",
  "sources": [
    {
      "document_id": 1,
      "filename": "document.pdf",
      "chunk_index": 2,
      "content": "...the chunk of text that was retrieved...",
      "similarity": 0.8123
    }
  ]
}
```

## Configuration

All configuration is via environment variables (or `.env`) — see `.env.example`:

| Variable          | Default                                                      | Description                                    |
|-------------------|---------------------------------------------------------------|-------------------------------------------------|
| `ANTHROPIC_API_KEY` | *(required)*                                                | Your Anthropic API key                          |
| `ANTHROPIC_MODEL`   | `claude-sonnet-5`                                           | Claude model used to answer questions           |
| `DATABASE_URL`      | `postgresql://raguser:ragpassword@localhost:5432/ragdocs`  | Postgres connection string                      |
| `EMBEDDING_MODEL`   | `all-MiniLM-L6-v2`                                          | sentence-transformers model (must stay 384-dim, or update `sql/init.sql`) |
| `CHUNK_SIZE`        | `500`                                                        | Characters per chunk                            |
| `CHUNK_OVERLAP`     | `50`                                                          | Overlap between consecutive chunks              |
| `TOP_K`             | `4`                                                           | Number of chunks retrieved per question         |

## Design notes / trade-offs

- **Chunking is character-based, not token-based.** This keeps the implementation
  dependency-free (no tokenizer needed just to chunk text). ~500 characters is a
  reasonable proxy for a few hundred tokens of English prose.
- **Cosine similarity via pgvector's `<=>` operator**, with an HNSW index. HNSW was
  chosen over IVFFlat because it doesn't require pre-existing data to build a
  well-clustered index, which matters for a demo app whose table starts empty.
- **Embeddings are local and free.** Only the final answer-generation step calls an
  external API (Anthropic). This means ingestion works completely offline, and you can
  swap in a different local embedding model without touching your Anthropic usage.
- **The LLM is explicitly instructed to cite excerpt numbers** and to say when the
  context doesn't contain enough information, rather than guessing — the system prompt
  in `app/llm.py` encodes this.

## Caveats

- The `all-MiniLM-L6-v2` model has a limited context window per chunk (~256 tokens),
  so very long chunks may get truncated during embedding. The default chunk size (500
  characters, well under that) is chosen accordingly.
- PDF text extraction via `pypdf` works well for text-based PDFs but will not extract
  text from scanned/image-only PDFs (no OCR is performed).
- This is a portfolio/demo project — there's no authentication, multi-tenancy, or rate
  limiting. Don't expose it to the public internet as-is.
