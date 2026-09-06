"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    content_type: str | None
    uploaded_at: datetime
    num_chunks: int


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    num_chunks: int
    message: str = "Document ingested successfully."


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about your documents.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override the number of chunks to retrieve."
    )


class SourceChunk(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    content: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str
    anthropic_configured: bool
