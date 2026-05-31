"""
Embedding providers for the RAG financial knowledge base.

Provides two embedding model wrappers:
- EmbeddingModel: OpenAI API-based embeddings (text-embedding-3-small)
- LocalEmbeddingModel: Local sentence-transformers (BGE-M3) for offline use

Both classes share a common interface: embed_texts, embed_query, get_dimension.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU Cache for embeddings
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_MAX = 10_000


class _EmbeddingCache:
    """Thread-unsafe but simple LRU cache mapping text hash -> vector."""

    def __init__(self, maxsize: int = _DEFAULT_CACHE_MAX) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        key = self._hash(text)
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, text: str, vector: list[float]) -> None:
        key = self._hash(text)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = vector
            return
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)  # evict oldest
        self._store[key] = vector

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store), "hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


# Module-level singleton cache
_embedding_cache = _EmbeddingCache()


# ---------------------------------------------------------------------------
# OpenAI Embedding Model
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Wrapper for OpenAI embedding models.

    Provides batch embedding, single query embedding, and LRU caching
    with automatic retry for rate-limit errors.

    Attributes:
        model_name: The OpenAI model identifier.
    """

    # OpenAI embedding dimension mapping
    _DIMENSION_MAP: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
        max_retries: int = 5,
    ) -> None:
        """Initialize OpenAI embedding client.

        Args:
            model_name: OpenAI embedding model name.
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
            batch_size: Maximum texts per API call.
            max_retries: Maximum retry attempts for transient errors.

        Raises:
            ImportError: If the openai package is not installed.
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for EmbeddingModel. "
                "Install it with: pip install openai"
            ) from exc

        self.model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]

        self._client = OpenAI(**kwargs)
        self._dimension = self._DIMENSION_MAP.get(model_name, 1536)
        logger.info(
            "Initialized EmbeddingModel with model=%s, dimension=%d",
            model_name,
            self._dimension,
        )

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI API with exponential backoff retry."""
        import random

        last_exc: Exception | None = None
        backoff = 1.0

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                last_exc = exc
                error_str = str(exc).lower()
                is_retryable = any(
                    kw in error_str
                    for kw in ["rate", "429", "timeout", "overloaded", "503"]
                )
                if not is_retryable or attempt == self._max_retries:
                    break
                jitter = random.uniform(0, 0.5 * backoff)
                sleep_time = backoff + jitter
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s – retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
                backoff *= 2

        raise RuntimeError(
            f"OpenAI embedding failed after {self._max_retries} attempts"
        ) from last_exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using OpenAI API.

        Uses LRU cache to avoid redundant API calls. Processes texts
        in batches for efficiency.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        # Check cache for each text
        results: list[list[float] | None] = [_embedding_cache.get(t) for t in texts]
        uncached_indices = [i for i, v in enumerate(results) if v is None]

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            logger.debug(
                "Cache miss for %d texts, fetching from OpenAI", len(uncached_texts)
            )

            # Process in batches
            all_embeddings: list[list[float]] = []
            for start in range(0, len(uncached_texts), self._batch_size):
                batch = uncached_texts[start : start + self._batch_size]
                all_embeddings.extend(self._embed_batch_with_retry(batch))

            # Update cache and results
            for idx, vec in zip(uncached_indices, all_embeddings):
                results[idx] = vec
                _embedding_cache.put(texts[idx], vec)

        return [v for v in results]  # type: ignore[misc]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Uses cache if available.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        cached = _embedding_cache.get(query)
        if cached is not None:
            return cached
        vector = self._embed_batch_with_retry([query])[0]
        _embedding_cache.put(query, vector)
        return vector

    def get_dimension(self) -> int:
        """Return the embedding dimension for the current model.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self._dimension


# ---------------------------------------------------------------------------
# Local Embedding Model (sentence-transformers)
# ---------------------------------------------------------------------------


class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers.

    Loads a pre-trained model (default: BAAI/bge-m3) for offline embedding.
    No API calls are made; all computation happens locally.

    Attributes:
        model_name: The HuggingFace model identifier.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the local embedding model.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            batch_size: Batch size for encoding.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required for "
                "LocalEmbeddingModel. Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._batch_size = batch_size

        logger.info("Loading local embedding model: %s ...", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        # 获取模型的 embedding 维度
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Local embedding model loaded: %s (dim=%d, device=%s)",
            model_name,
            self._dimension,
            self._model.device,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using the local model.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化以使用余弦相似度
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string using the local model.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """Return the embedding dimension of the local model.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self._dimension


# ---------------------------------------------------------------------------
# Factory & helpers
# ---------------------------------------------------------------------------


def embedding_model_factory(
    provider: str = "openai",
    **kwargs: Any,
) -> EmbeddingModel | LocalEmbeddingModel:
    """Factory that returns an embedding model by provider name.

    Supported providers:
        - 'openai' → EmbeddingModel (OpenAI API)
        - 'local' / 'bge-m3' → LocalEmbeddingModel (sentence-transformers)

    Args:
        provider: Provider name.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An instance of the requested embedding model.
    """
    name = provider.strip().lower()
    if name in ("openai",):
        return EmbeddingModel(**kwargs)
    if name in ("local", "bge-m3", "sentence-transformers"):
        return LocalEmbeddingModel(**kwargs)
    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        f"Supported: openai, local, bge-m3"
    )


def clear_cache() -> None:
    """Clear the global embedding cache (useful in tests)."""
    _embedding_cache.clear()


def cache_stats() -> dict[str, int]:
    """Return current cache statistics."""
    return _embedding_cache.stats()
