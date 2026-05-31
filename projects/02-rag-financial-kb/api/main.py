"""
FastAPI application for the RAG Financial Knowledge Base.

Provides REST API endpoints for querying the RAG pipeline,
ingesting documents, and checking system health/stats.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project imports (adjust path as needed)
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.document_loader import Document, load_directory, load_file
from src.embedding import EmbeddingProvider, embedding_factory
from src.rag_engine import RAGEngine, RAGResponse, build_rag_engine
from src.reranker import SimpleReranker, reranker_factory
from src.retriever import retriever_factory
from src.text_splitter import Chunk, split_documents
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------
DEFAULT_VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH", str(_PROJECT_ROOT / "data" / "vector_store.pkl")
)
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_RETRIEVER = os.getenv("RETRIEVER", "similarity")
DEFAULT_RERANKER = os.getenv("RERANKER", "simple")

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., description="Natural language question about finance")
    top_k: int = Field(5, ge=1, le=50, description="Number of documents to retrieve")
    use_reranker: bool = Field(True, description="Whether to apply reranking")


class IngestFromPathRequest(BaseModel):
    """Request body for ingesting from a directory/file path."""

    path: str = Field(..., description="Directory or file path to ingest")
    chunk_size: int = Field(500, ge=50, le=5000, description="Target chunk size")
    strategy: str = Field(
        "recursive",
        description="Splitting strategy: recursive, semantic, or financial",
    )


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str
    vector_store_count: int
    embedding_provider: str


class StatsResponse(BaseModel):
    """Response for the /stats endpoint."""

    total_documents: int
    embedding_provider: str
    embedding_cache_stats: dict[str, int]
    vector_store_path: str


class IngestResponse(BaseModel):
    """Response for the /ingest endpoint."""

    message: str
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Application state (singleton-ish via module globals)
# ---------------------------------------------------------------------------
_vector_store: VectorStore | None = None
_embedding_provider: EmbeddingProvider | None = None
_rag_engine: RAGEngine | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = embedding_factory(DEFAULT_EMBEDDING_PROVIDER)
    return _embedding_provider


def _get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        vs = _get_vector_store()
        ep = _get_embedding_provider()
        retriever = retriever_factory(DEFAULT_RETRIEVER, vs, ep)
        reranker = reranker_factory(DEFAULT_RERANKER)
        _rag_engine = RAGEngine(
            vector_store=vs,
            embedding_provider=ep,
            retriever=retriever,
            reranker=reranker,
            llm_model=DEFAULT_LLM_MODEL,
        )
    return _rag_engine


def _rebuild_rag_engine() -> RAGEngine:
    """Rebuild the RAG engine after index changes."""
    global _rag_engine
    _rag_engine = None
    return _get_rag_engine()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Financial Knowledge Base API",
    description="Retrieval-Augmented Generation API for financial documents",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize components on startup."""
    logger.info("Starting RAG Financial KB API")
    # Pre-initialize embedding provider
    _get_embedding_provider()
    # Try to load existing vector store
    vs_path = Path(DEFAULT_VECTOR_STORE_PATH)
    if vs_path.exists():
        global _vector_store
        try:
            _vector_store = VectorStore.load(str(vs_path))
            logger.info("Loaded existing vector store from %s", vs_path)
        except Exception as exc:
            logger.warning("Could not load vector store: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    vs = _get_vector_store()
    return HealthResponse(
        status="healthy",
        vector_store_count=vs.count(),
        embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get vector store and system statistics."""
    from src.embedding import cache_stats

    vs = _get_vector_store()
    return StatsResponse(
        total_documents=vs.count(),
        embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
        embedding_cache_stats=cache_stats(),
        vector_store_path=DEFAULT_VECTOR_STORE_PATH,
    )


@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest) -> RAGResponse:
    """Query the RAG pipeline with a natural language question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vs = _get_vector_store()
    if vs.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Please ingest documents first.",
        )

    try:
        engine = _get_rag_engine()
        response = engine.query(
            question=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
        return response
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(exc)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    file: UploadFile | None = File(None),
    request: IngestFromPathRequest | None = None,
) -> IngestResponse:
    """
    Ingest documents into the vector store.

    Supports two modes:
    - Upload a file directly
    - Provide a directory/file path via JSON body
    """
    documents: list[Document] = []

    if file is not None:
        # Save uploaded file to temp location
        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            loaded = load_file(tmp_path)
            documents.extend(loaded)
        finally:
            os.unlink(tmp_path)

    elif request is not None:
        target = Path(request.path)
        if not target.exists():
            raise HTTPException(
                status_code=400, detail=f"Path does not exist: {request.path}"
            )
        if target.is_dir():
            documents.extend(load_directory(request.path))
        else:
            documents.extend(load_file(request.path))
    else:
        raise HTTPException(
            status_code=400,
            detail="Either provide a file upload or a path in the request body",
        )

    if not documents:
        return IngestResponse(
            message="No documents were loaded from the input",
            documents_loaded=0,
            chunks_created=0,
            chunks_indexed=0,
        )

    # Split documents into chunks
    strategy = (request.strategy if request else "recursive") or "recursive"
    chunk_size = (request.chunk_size if request else 500) or 500
    chunks = split_documents(documents, strategy=strategy, chunk_size=chunk_size)

    if not chunks:
        return IngestResponse(
            message="Documents loaded but produced no chunks",
            documents_loaded=len(documents),
            chunks_created=0,
            chunks_indexed=0,
        )

    # Embed and add to vector store
    ep = _get_embedding_provider()
    texts = [c.content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Embed in batches
    embeddings = ep.cached_embed(texts)

    vs = _get_vector_store()
    ids = vs.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

    # Rebuild RAG engine to pick up new data
    _rebuild_rag_engine()

    logger.info(
        "Ingested %d documents → %d chunks → %d indexed",
        len(documents),
        len(chunks),
        len(ids),
    )

    return IngestResponse(
        message="Successfully ingested documents",
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=len(ids),
    )
