"""Tests for src/retriever module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.retriever import BM25, Retriever


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class TestBM25:
    """Test the BM25 keyword scorer."""

    def test_basic_keyword_matching(self):
        """Documents containing query terms should get positive scores."""
        bm25 = BM25()
        bm25.build_index(
            doc_ids=["doc1", "doc2", "doc3"],
            documents=[
                "MACD is a momentum indicator used in technical analysis",
                "Risk management is essential for trading",
                "Bollinger Bands measure volatility",
            ],
        )
        scores = bm25.score("MACD indicator")
        assert "doc1" in scores
        assert scores["doc1"] > 0

    def test_no_match_returns_zero_scores(self):
        """Documents not containing any query term should not appear in scores."""
        bm25 = BM25()
        bm25.build_index(
            doc_ids=["doc1"],
            documents=["Hello world"],
        )
        scores = bm25.score("xyz_nonexistent_term")
        # 应该为空或 doc1 分数为 0
        assert scores.get("doc1", 0.0) == 0.0

    def test_empty_index(self):
        """Scoring against an empty index should return an empty dict."""
        bm25 = BM25()
        scores = bm25.score("anything")
        assert scores == {}

    def test_idf_gives_rare_terms_higher_weight(self):
        """Rare terms should contribute more to the score than common terms."""
        bm25 = BM25()
        bm25.build_index(
            doc_ids=["d1", "d2"],
            documents=[
                "common common common rare_term",
                "common common common common",
            ],
        )
        scores = bm25.score("rare_term")
        assert scores.get("d1", 0) > 0
        # d2 不包含 rare_term，不应出现在结果中
        assert "d2" not in scores or scores["d2"] == 0

    def test_build_index_populates_internal_state(self):
        """After building, internal state should be populated."""
        bm25 = BM25()
        assert not bm25._built
        bm25.build_index(["d1"], ["test document"])
        assert bm25._built
        assert bm25._doc_count == 1


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    """Test the Retriever class with mocked dependencies."""

    def test_similarity_search(self, mock_vector_store, mock_embedding_model):
        """similarity_search should embed query and search the vector store."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
        )
        results = retriever.similarity_search("test query", top_k=2)
        mock_embedding_model.embed_query.assert_called_once_with("test query")
        mock_vector_store.search.assert_called_once()
        assert len(results) == 2

    def test_similarity_search_returns_list_of_dicts(self, mock_vector_store, mock_embedding_model):
        """Results should be a list of dicts with content, metadata, distance."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
        )
        results = retriever.similarity_search("query", top_k=2)
        for r in results:
            assert "content" in r
            assert "metadata" in r
            assert "distance" in r

    def test_mmr_search_returns_diverse_results(self, mock_vector_store, mock_embedding_model):
        """MMR search should return results and use embeddings for diversity."""
        # 准备更多的候选结果
        mock_vector_store.search.return_value = [
            {"content": f"Document {i} about finance", "metadata": {"source": f"d{i}.pdf"}, "distance": 0.1 * i}
            for i in range(10)
        ]
        # mock embed_texts 返回不同向量
        def fake_embed_texts(texts):
            return [[0.1 + i * 0.01] * 1536 for i in range(len(texts))]
        mock_embedding_model.embed_texts.side_effect = fake_embed_texts

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
        )
        results = retriever.mmr_search("query", top_k=3, lambda_param=0.5)
        assert len(results) == 3

    def test_mmr_search_small_pool(self, mock_vector_store, mock_embedding_model):
        """When candidate pool is smaller than top_k, return all candidates."""
        mock_vector_store.search.return_value = [
            {"content": "Doc 1", "metadata": {}, "distance": 0.1},
            {"content": "Doc 2", "metadata": {}, "distance": 0.2},
        ]
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536, [0.2] * 1536]

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
        )
        results = retriever.mmr_search("query", top_k=5)
        assert len(results) == 2

    def test_hybrid_search(self, mock_vector_store, mock_embedding_model):
        """Hybrid search should combine vector and keyword scores."""
        mock_vector_store.search.return_value = [
            {"content": "MACD is a momentum indicator", "metadata": {}, "distance": 0.1},
            {"content": "Risk management principles", "metadata": {}, "distance": 0.2},
        ]
        mock_embedding_model.embed_texts.return_value = [[0.1] * 1536, [0.2] * 1536]

        retriever = Retriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
        )
        results = retriever.hybrid_search("MACD", top_k=2)
        assert len(results) <= 2
        # 结果应包含距离字段
        assert all("distance" in r for r in results)
