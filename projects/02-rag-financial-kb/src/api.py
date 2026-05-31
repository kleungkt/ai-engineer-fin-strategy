"""
FastAPI application for the RAG financial knowledge base.

Provides REST endpoints for querying the knowledge base, ingesting documents,
viewing stats, direct search, and health checks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag_engine import RAGEngine, RAGResponse
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.embedding import EmbeddingModel
from src.document_loader import load_directory
from src.text_splitter import split_documents

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., description="User's natural language question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")
    use_rewrite: bool = Field(default=True, description="Whether to rewrite the query via LLM")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    directory: str = Field(..., description="Directory path containing documents to ingest")


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    status: str
    documents_loaded: int
    chunks_created: int


class StatsResponse(BaseModel):
    """Response body for GET /stats."""

    total_chunks: int
    collection_name: str
    persist_dir: str


class SearchItem(BaseModel):
    """A single search result for GET /search."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float = 0.0


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes registered.
    """
    app = FastAPI(
        title="RAG Financial Knowledge Base",
        description="API for querying a financial knowledge base using RAG",
        version="0.1.0",
    )

    # CORS 中间件 — 允许前端跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # 模块级组件（延迟初始化）
    # ------------------------------------------------------------------
    _state: dict[str, Any] = {
        "rag_engine": None,
        "retriever": None,
        "vector_store": None,
        "embedding_model": None,
    }

    def _get_vector_store() -> VectorStore:
        """Lazy-initialize the vector store."""
        if _state["vector_store"] is None:
            from config.settings import settings

            _state["vector_store"] = VectorStore(
                persist_dir=settings.CHROMA_PERSIST_DIR,
                collection_name="financial_kb",
            )
        return _state["vector_store"]

    def _get_embedding_model() -> EmbeddingModel:
        """Lazy-initialize the embedding model."""
        if _state["embedding_model"] is None:
            from config.settings import settings

            _state["embedding_model"] = EmbeddingModel(
                model_name=settings.EMBEDDING_MODEL,
            )
        return _state["embedding_model"]

    def _get_retriever() -> Retriever:
        """Lazy-initialize the retriever."""
        if _state["retriever"] is None:
            _state["retriever"] = Retriever(
                vector_store=_get_vector_store(),
                embedding_model=_get_embedding_model(),
            )
        return _state["retriever"]

    def _get_rag_engine() -> RAGEngine:
        """Lazy-initialize the RAG engine."""
        if _state["rag_engine"] is None:
            from config.settings import settings

            _state["rag_engine"] = RAGEngine(
                retriever=_get_retriever(),
                llm_model=settings.OPENAI_MODEL,
            )
        return _state["rag_engine"]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    @app.post("/query", response_model=RAGResponse)
    async def query(req: QueryRequest) -> RAGResponse:
        """RAG query endpoint: retrieve + generate answer."""
        try:
            engine = _get_rag_engine()
            response = engine.query(
                question=req.question,
                top_k=req.top_k,
                use_rewrite=req.use_rewrite,
            )
            return response
        except Exception as exc:
            logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest) -> IngestResponse:
        """Ingest documents from a directory into the knowledge base."""
        dir_path = Path(req.directory)
        if not dir_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {req.directory}",
            )

        try:
            # 1. 加载文档
            docs = load_directory(dir_path)
            if not docs:
                return IngestResponse(
                    status="no_documents_found",
                    documents_loaded=0,
                    chunks_created=0,
                )

            # 2. 分块
            from config.settings import settings

            chunks = split_documents(
                docs,
                strategy="recursive",
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            )

            # 3. Embedding
            embedding_model = _get_embedding_model()
            texts = [c.content for c in chunks]
            embeddings = embedding_model.embed_texts(texts)

            # 4. 存入向量数据库
            store = _get_vector_store()
            added = store.add_chunks(chunks, embeddings)
            store.persist()

            return IngestResponse(
                status="success",
                documents_loaded=len(docs),
                chunks_created=added,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Ingestion failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        """Return knowledge base statistics."""
        try:
            store = _get_vector_store()
            return StatsResponse(
                total_chunks=store.get_count(),
                collection_name=store._collection_name,
                persist_dir=store._persist_dir,
            )
        except Exception as exc:
            logger.exception("Stats retrieval failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/search", response_model=list[SearchItem])
    async def search(
        query: str = Query(..., description="Search query string"),
        top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    ) -> list[SearchItem]:
        """Direct vector search (no LLM generation)."""
        try:
            retriever = _get_retriever()
            results = retriever.similarity_search(query, top_k=top_k)
            return [
                SearchItem(
                    content=r.get("content", ""),
                    metadata=r.get("metadata", {}),
                    distance=r.get("distance", 0.0),
                )
                for r in results
            ]
        except Exception as exc:
            logger.exception("Search failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ---------------------------------------------------------------------------
# Module-level app instance
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
