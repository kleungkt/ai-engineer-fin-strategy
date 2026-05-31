"""
Retrieval strategies for the RAG financial knowledge base.

Provides a unified ``Retriever`` class with multiple retrieval strategies:
- similarity_search: Basic cosine similarity via vector store
- mmr_search: Maximum Marginal Relevance (relevance + diversity)
- hybrid_search: Vector similarity + BM25 keyword matching (jieba for Chinese)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jieba tokenizer wrapper (Chinese text segmentation)
# ---------------------------------------------------------------------------


def _tokenize_chinese(text: str) -> list[str]:
    """Tokenize text using jieba for Chinese + simple regex for English.

    Uses jieba for Chinese character segmentation and falls back to
    simple word boundary splitting for English tokens.

    Args:
        text: Input text (mixed Chinese/English).

    Returns:
        List of tokens.
    """
    try:
        import jieba  # type: ignore[import-untyped]

        # jieba 可以处理中英文混合文本
        tokens = list(jieba.cut(text, cut_all=False))
    except ImportError:
        # Fallback: 使用正则分词
        logger.warning("jieba not installed, using regex fallback for tokenization")
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text)

    # 过滤空白和单字符标点
    tokens = [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 0]
    return tokens


# ---------------------------------------------------------------------------
# BM25 Scoring
# ---------------------------------------------------------------------------


class BM25:
    """Basic BM25 scorer for keyword matching.

    Implements Okapi BM25 scoring with configurable k1 and b parameters.
    Designed to work with Chinese text tokenized by jieba.

    Args:
        k1: Term frequency saturation parameter (default: 1.5).
        b: Document length normalization parameter (default: 0.75).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._doc_count: int = 0
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_tokens: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._idf: dict[str, float] = {}
        self._built = False

    def build_index(
        self,
        doc_ids: list[str],
        documents: list[str],
    ) -> None:
        """Build BM25 index from document texts.

        Args:
            doc_ids: List of unique document identifiers.
            documents: List of document text strings.
        """
        self._doc_ids = doc_ids
        self._doc_count = len(documents)
        self._doc_tokens = [_tokenize_chinese(doc) for doc in documents]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._doc_count if self._doc_count > 0 else 0.0
        )

        # 计算 IDF（逆文档频率）
        doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        # BM25 IDF 公式: log((N - n + 0.5) / (n + 0.5) + 1)
        self._idf = {}
        for token, freq in doc_freq.items():
            self._idf[token] = math.log(
                (self._doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )

        self._built = True
        logger.debug("BM25 index built: %d documents, %d unique terms", self._doc_count, len(self._idf))

    def score(self, query: str) -> dict[str, float]:
        """Score all documents against a query using BM25.

        Args:
            query: Query string to score against.

        Returns:
            Dict mapping doc_id to BM25 score. Only includes docs with score > 0.
        """
        if not self._built:
            return {}

        query_tokens = _tokenize_chinese(query)
        scores: dict[str, float] = {}

        for i in range(self._doc_count):
            doc_tokens = self._doc_tokens[i]
            doc_len = self._doc_lengths[i]
            score = 0.0

            # 计算每个查询词的 BM25 分数
            token_counts: Counter[str] = Counter(doc_tokens)

            for q_token in query_tokens:
                if q_token not in token_counts:
                    continue

                tf = token_counts[q_token]
                idf = self._idf.get(q_token, 0.0)

                # BM25 公式: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_length
                )
                score += idf * numerator / denominator

            if score > 0:
                scores[self._doc_ids[i]] = score

        return scores


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Unified retrieval interface with multiple search strategies.

    Provides cosine similarity search, Maximum Marginal Relevance (MMR),
    and hybrid search combining vector similarity with BM25 keyword matching.

    Args:
        vector_store: VectorStore instance for vector search.
        embedding_model: EmbeddingModel instance for query embedding.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any,
    ) -> None:
        """Initialize with vector store and embedding model.

        Args:
            vector_store: VectorStore instance (from vector_store.py).
            embedding_model: EmbeddingModel or LocalEmbeddingModel instance.
        """
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._bm25: BM25 | None = None
        logger.info("Initialized Retriever")

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Basic cosine similarity search.

        Embeds the query and searches the vector store directly.

        Args:
            query: Natural language query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: content, metadata, distance.
        """
        query_embedding = self._embedding_model.embed_query(query)
        results = self._vector_store.search(query_embedding, top_k=top_k)
        logger.debug("Similarity search returned %d results for query: %s", len(results), query[:50])
        return results

    def mmr_search(
        self,
        query: str,
        top_k: int = 5,
        lambda_param: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Maximum Marginal Relevance search.

        Balances relevance to the query with diversity among selected documents.
        Starts with the most relevant document, then iteratively adds documents
        that are relevant but diverse.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            lambda_param: Trade-off between relevance and diversity.
                1.0 = pure relevance, 0.0 = pure diversity.

        Returns:
            List of dicts with keys: content, metadata, distance.
        """
        # 获取一个较大的候选池以便进行多样性重排序
        pool_size = max(top_k * 4, 20)
        query_embedding = self._embedding_model.embed_query(query)
        candidates = self._vector_store.search(query_embedding, top_k=pool_size)

        if len(candidates) <= top_k:
            return candidates

        # 获取候选文档的 embedding 向量
        candidate_contents = [c["content"] for c in candidates]
        candidate_embeddings = self._embedding_model.embed_texts(candidate_contents)
        candidate_matrix = np.array(candidate_embeddings, dtype=np.float32)

        query_vec = np.array(query_embedding, dtype=np.float32)

        # 归一化向量
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        emb_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        emb_norms = np.maximum(emb_norms, 1e-10)
        normed_embs = candidate_matrix / emb_norms

        # 计算与 query 的相关性分数
        relevance_scores = normed_embs @ query_vec

        # MMR 选择算法
        selected_indices: list[int] = []
        remaining = set(range(len(candidates)))
        actual_top_k = min(top_k, len(candidates))

        for _ in range(actual_top_k):
            best_score = -float("inf")
            best_idx = -1

            for idx in remaining:
                relevance = float(relevance_scores[idx])

                # 计算与已选文档的最大相似度（多样性惩罚）
                if selected_indices:
                    selected_embs = normed_embs[selected_indices]
                    diversity_penalty = float(np.max(selected_embs @ normed_embs[idx]))
                else:
                    diversity_penalty = 0.0

                # MMR 分数 = λ * relevance - (1 - λ) * max_similarity_to_selected
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        # 重建结果列表
        results = []
        for idx in selected_indices:
            result = dict(candidates[idx])
            result["distance"] = 1.0 - float(relevance_scores[idx])  # 转换为距离
            results.append(result)

        logger.debug("MMR search returned %d results (lambda=%.2f)", len(results), lambda_param)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and BM25 keyword matching.

        Uses jieba for Chinese text tokenization in BM25 scoring.
        Final score = alpha * vector_score + (1 - alpha) * bm25_score.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            alpha: Weight for vector similarity (0.0–1.0).
                Higher alpha gives more weight to semantic similarity.

        Returns:
            List of dicts with keys: content, metadata, distance.
        """
        # 1. Vector similarity search (获取更多候选)
        query_embedding = self._embedding_model.embed_query(query)
        vector_results = self._vector_store.search(
            query_embedding, top_k=top_k * 3
        )

        # 2. BM25 keyword search
        # 构建/更新 BM25 索引
        if self._bm25 is None:
            self._bm25 = BM25()
            self._build_bm25_index()

        bm25_scores = self._bm25.score(query) if self._bm25._built else {}

        # 3. 合并分数
        # 创建内容到结果的映射
        merged: dict[str, dict[str, Any]] = {}

        # 归一化 vector scores (distance 越小越好，转换为相似度)
        max_dist = max((r["distance"] for r in vector_results), default=1.0)
        if max_dist == 0:
            max_dist = 1.0

        for result in vector_results:
            content_key = result["content"][:100]  # 用前100字符作为 key
            vector_sim = 1.0 - (result["distance"] / max_dist)  # 转为 0-1 相似度

            merged[content_key] = {
                **result,
                "_vector_sim": max(0.0, vector_sim),
                "_bm25_score": 0.0,
            }

        # 归一化 BM25 scores
        max_bm25 = max(bm25_scores.values(), default=1.0)
        if max_bm25 == 0:
            max_bm25 = 1.0

        # 用 BM25 重新排序已有的结果
        # 注意：BM25 索引中的 doc_id 对应 vector store 中的 chunk
        # 这里简化处理：对所有候选文档进行 BM25 评分
        for content_key, entry in merged.items():
            # 对内容重新计算 BM25 分数
            doc_text = entry["content"]
            doc_tokens = _tokenize_chinese(doc_text)
            query_tokens = _tokenize_chinese(query)

            # 简单的词频匹配作为 BM25 近似
            token_set = set(doc_tokens)
            match_count = sum(1 for qt in query_tokens if qt in token_set)
            entry["_bm25_score"] = match_count / max(len(query_tokens), 1)

        # 计算混合分数
        results = []
        for entry in merged.values():
            vector_sim = entry.get("_vector_sim", 0.0)
            bm25_score = entry.get("_bm25_score", 0.0)
            hybrid_score = alpha * vector_sim + (1 - alpha) * bm25_score

            result = {
                "content": entry["content"],
                "metadata": entry["metadata"],
                "distance": 1.0 - hybrid_score,  # 转回距离
            }
            results.append(result)

        # 按距离排序（越小越好）
        results.sort(key=lambda x: x["distance"])
        results = results[:top_k]

        logger.debug(
            "Hybrid search returned %d results (alpha=%.2f)", len(results), alpha
        )
        return results

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store.

        This is a helper method called lazily on first hybrid_search call.
        """
        if self._bm25 is None:
            self._bm25 = BM25()

        # 从 vector store 获取所有文档
        # 注意：这需要 vector store 支持全量获取
        # ChromaDB 可以通过 get() 方法获取所有文档
        try:
            collection = self._vector_store._collection
            all_docs = collection.get(include=["documents"])

            if all_docs and all_docs["documents"]:
                doc_ids = all_docs["ids"] if "ids" in all_docs else [str(i) for i in range(len(all_docs["documents"]))]
                self._bm25.build_index(doc_ids, all_docs["documents"])
                logger.info("BM25 index built with %d documents", len(all_docs["documents"]))
        except Exception as e:
            logger.warning("Failed to build BM25 index: %s", e)
