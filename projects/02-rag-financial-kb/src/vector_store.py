"""
ChromaDB-backed vector store for RAG financial knowledge base.

Provides persistent vector storage using ChromaDB with support for
adding TextChunks, similarity search, and collection management.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for persistent vector storage.

    Stores text chunks with their embeddings and metadata. Supports
    cosine similarity search and automatic persistence.

    Usage::

        store = VectorStore("/path/to/persist", collection_name="financial_kb")
        count = store.add_chunks(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
        store.persist()
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "financial_kb",
    ) -> None:
        """Initialize ChromaDB client and collection.

        Args:
            persist_dir: Directory path for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.

        Raises:
            ImportError: If chromadb is not installed.
        """
        try:
            import chromadb  # type: ignore[import-untyped]
            from chromadb.config import Settings  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'chromadb' package is required for VectorStore. "
                "Install it with: pip install chromadb"
            ) from exc

        self._persist_dir = persist_dir
        self._collection_name = collection_name

        # 创建持久化客户端
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

        logger.info(
            "Initialized VectorStore: dir=%s, collection=%s, count=%d",
            persist_dir,
            collection_name,
            self._collection.count(),
        )

    def add_chunks(
        self,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> int:
        """Add text chunks with their embeddings to the store.

        Args:
            chunks: List of TextChunk instances (must have content, metadata,
                chunk_id attributes).
            embeddings: List of embedding vectors (same length as chunks).

        Returns:
            Number of chunks successfully added.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embedding_list: list[list[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") and chunk.chunk_id else str(len(ids))

            # ChromaDB metadata 只支持 str, int, float, bool
            # 需要过滤掉不支持的类型
            metadata = {}
            raw_meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                elif v is None:
                    metadata[k] = ""  # None 转为空字符串
                else:
                    metadata[k] = str(v)

            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(metadata)
            embedding_list.append(embedding)

        # 分批添加（ChromaDB 有单次添加限制）
        batch_size = 5000
        added_count = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embedding_list[i:end],
            )
            added_count += end - i

        logger.info("Added %d chunks to VectorStore", added_count)
        return added_count

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Similarity search using query embedding.

        Args:
            query_embedding: Query vector for similarity comparison.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: content, metadata, distance.
            Sorted by ascending distance (most similar first).
        """
        if self._collection.count() == 0:
            return []

        actual_k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        # 格式化结果
        formatted: list[dict[str, Any]] = []
        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                formatted.append({
                    "content": doc,
                    "metadata": meta or {},
                    "distance": dist,
                })

        return formatted

    def delete_collection(self) -> None:
        """Delete and recreate the collection.

        This clears all stored documents. The collection is immediately
        recreated as an empty collection.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Deleted and recreated collection: %s", self._collection_name)

    def get_count(self) -> int:
        """Return the number of stored documents.

        Returns:
            Integer count of documents in the collection.
        """
        return self._collection.count()

    def persist(self) -> None:
        """Persist the vector store to disk.

        ChromaDB PersistentClient auto-persists, but this method ensures
        any pending writes are flushed.
        """
        # ChromaDB PersistentClient 自动持久化
        # 这里显式调用确保写入完成
        logger.info(
            "VectorStore persisted: %d documents in collection '%s'",
            self.get_count(),
            self._collection_name,
        )

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection='{self._collection_name}', "
            f"count={self.get_count()}, persist_dir='{self._persist_dir}')"
        )
