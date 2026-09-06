"""FastAPI application: RAG-over-your-docs chat backend.

Endpoints:
    GET  /              -> static single-page frontend
    GET  /health        -> service + dependency health check
    POST /documents     -> upload + ingest a .txt/.md/.pdf file
    GET  /documents     -> list ingested documents
    POST /chat          -> ask a question, get an answer grounded in your docs
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import get_settings
from .extraction import UnsupportedFileTypeError
from .ingestion import EmptyDocumentError, ingest_document
from .llm import answer_question
from .retrieval import retrieve_relevant_chunks
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    DocumentUploadResponse,
    HealthResponse,
    SourceChunk,
)

logger = logging.getLogger("rag_docs_chat")
logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="rag-docs-chat",
    description="Upload documents, ask questions, get answers grounded in your own data.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    db_ok = db.check_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else "unreachable",
        embedding_model=settings.embedding_model,
        anthropic_configured=bool(settings.anthropic_api_key),
    )


@app.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20MB).")

    try:
        result = ingest_document(filename, file.content_type, data)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface DB/embedding errors as 500s
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return DocumentUploadResponse(**result)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    try:
        rows = db.list_documents()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=f"Could not list documents: {exc}") from exc
    return [DocumentOut(**row) for row in rows]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        chunks = retrieve_relevant_chunks(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    try:
        answer = answer_question(request.question, chunks)
    except RuntimeError as exc:
        # Missing API key or upstream Anthropic error -- surface as 502/400
        # rather than a generic 500 so the frontend can show a clear message.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = [
        SourceChunk(
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            content=c["content"],
            similarity=round(float(c["similarity"]), 4),
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)
