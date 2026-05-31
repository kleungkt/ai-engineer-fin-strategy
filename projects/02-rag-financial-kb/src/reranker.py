"""
Reranker module for RAG financial knowledge base.

Provides reranking strategies to improve retrieval quality by reordering
documents based on more sophisticated relevance scoring.
"""

from abc import ABC, abstractmethod
import math
import re
from collections import Counter
from typing import Any


class RerankerBase(ABC):
    """Abstract base class for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank a list of documents based on relevance to the query.

        Args:
            query: The search query string.
            documents: List of document dicts, each with at least a 'content' key.
            top_k: Number of top documents to return after reranking.

        Returns:
            List of top_k documents sorted by relevance (highest first).
        """
        ...


class CrossEncoderReranker(RerankerBase):
    """
    Reranker using a sentence-transformers cross-encoder model.

    Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' by default.
    Falls back to SimpleReranker behavior if sentence-transformers is not installed.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._fallback: SimpleReranker | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self.model_name)
            self._available = True
        except ImportError:
            print(
                f"[Reranker] sentence-transformers not installed. "
                f"Falling back to SimpleReranker for '{self.model_name}'."
            )
            self._fallback = SimpleReranker()
        except Exception as exc:
            print(
                f"[Reranker] Failed to load model '{self.model_name}': {exc}. "
                f"Falling back to SimpleReranker."
            )
            self._fallback = SimpleReranker()

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank documents using cross-encoder scores, or fallback overlap scoring."""
        if not documents:
            return []

        if not self._available or self._fallback is not None:
            return self._fallback.rerank(query, documents, top_k)

        # Build query-document pairs for the cross-encoder
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # Get relevance scores from the cross-encoder
        scores: list[float] = self._model.predict(pairs)  # type: ignore[union-attr]

        # Attach scores and sort descending
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = dict(doc)
            scored_doc["rerank_score"] = float(score)
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_k]


class SimpleReranker(RerankerBase):
    """
    Lightweight reranker using BM25-style scoring with word overlap.

    No external ML dependencies required. Uses tokenisation, term frequency,
    inverse document frequency, and BM25 parameters to score document relevance.

    BM25 formula per term t:
        score(t) = IDF(t) * (tf(t, d) * (k1 + 1)) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 length normalisation parameter.
        """
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split text into alphanumeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        avgdl: float,
        idf: dict[str, float],
    ) -> float:
        """Compute BM25 score for a single document against the query."""
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in idf:
                continue
            term_tf = tf.get(token, 0)
            numerator = term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1.0))
            score += idf[token] * (numerator / denominator)

        return score

    def _compute_idf(self, all_doc_tokens: list[list[str]]) -> dict[str, float]:
        """Compute IDF for all terms across the document collection.

        IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        """
        n = len(all_doc_tokens)
        doc_freq: Counter[str] = Counter()
        for tokens in all_doc_tokens:
            unique = set(tokens)
            for token in unique:
                doc_freq[token] += 1

        idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        return idf

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank documents using BM25-style scoring."""
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        all_doc_tokens = [self._tokenize(doc.get("content", "")) for doc in documents]

        # Average document length
        total_len = sum(len(tokens) for tokens in all_doc_tokens)
        avgdl = total_len / max(len(all_doc_tokens), 1)

        # Compute IDF across the collection (query + document terms)
        idf = self._compute_idf(all_doc_tokens)

        # Score each document
        scored_docs = []
        for doc, doc_tokens in zip(documents, all_doc_tokens):
            bm25 = self._bm25_score(query_tokens, doc_tokens, avgdl, idf)

            # Blend with any pre-existing score if present
            existing_score = doc.get("score")
            if existing_score is not None:
                # Normalise existing score to [0, 1] range for blending
                normalised_existing = 1.0 / (1.0 + math.exp(-float(existing_score)))
                blended = 0.7 * bm25 + 0.3 * normalised_existing
            else:
                blended = bm25

            scored_doc = dict(doc)
            scored_doc["rerank_score"] = blended
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RERANKER_REGISTRY: dict[str, type[RerankerBase]] = {
    "cross_encoder": CrossEncoderReranker,
    "simple": SimpleReranker,
    "bm25": SimpleReranker,  # alias
}


def reranker_factory(name: str, **kwargs: Any) -> RerankerBase:
    """
    Create a RerankerBase instance by name.

    Supported names:
        - ``"cross_encoder"`` – CrossEncoderReranker (requires sentence-transformers)
        - ``"simple"``        – SimpleReranker (no ML dependencies)
        - ``"bm25"``          – alias for SimpleReranker

    Args:
        name: Reranker name (case-insensitive).
        **kwargs: Forwarded to the reranker constructor.

    Returns:
        A RerankerBase instance.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = name.lower().strip()
    if key not in _RERANKER_REGISTRY:
        available = ", ".join(sorted(_RERANKER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reranker '{name}'. Available: [{available}]"
        )
    return _RERANKER_REGISTRY[key](**kwargs)
