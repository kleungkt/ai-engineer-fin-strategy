"""Tests for src/rag_engine module."""

import pytest
from unittest.mock import MagicMock, patch

from src.rag_engine import RAGEngine, RAGResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_client(answer: str = "This is the generated answer."):
    """Create a mock OpenAI client that returns *answer*."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = answer
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# RAGResponse
# ---------------------------------------------------------------------------


class TestRAGResponse:
    """Test the RAGResponse Pydantic model."""

    def test_model_fields(self):
        resp = RAGResponse(
            answer="Test answer",
            sources=[{"content": "c", "source": "s", "score": 0.9}],
            rewritten_query="rewritten",
            confidence=0.85,
        )
        assert resp.answer == "Test answer"
        assert len(resp.sources) == 1
        assert resp.rewritten_query == "rewritten"
        assert resp.confidence == 0.85

    def test_defaults(self):
        resp = RAGResponse(answer="A")
        assert resp.sources == []
        assert resp.rewritten_query == ""
        assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# RAGEngine.query
# ---------------------------------------------------------------------------


class TestRAGEngineQuery:
    """Test RAGEngine.query with mocked retriever and LLM."""

    def test_query_returns_rag_response(self, mock_retriever):
        """query() should return a RAGResponse instance."""
        client = _make_openai_client("MACD 是一種動量指標。")
        engine = RAGEngine(
            retriever=mock_retriever,
            llm_model="gpt-4o-mini",
            openai_client=client,
        )
        response = engine.query("什麼是MACD？", top_k=3, use_rewrite=False)
        assert isinstance(response, RAGResponse)
        assert response.answer == "MACD 是一種動量指標。"
        assert len(response.sources) == 2

    def test_query_calls_retriever(self, mock_retriever):
        """query() should call retriever.similarity_search."""
        client = _make_openai_client()
        engine = RAGEngine(retriever=mock_retriever, openai_client=client)
        engine.query("test question", top_k=5, use_rewrite=False)
        mock_retriever.similarity_search.assert_called_once_with("test question", top_k=5)

    def test_query_with_rewrite(self, mock_retriever):
        """When use_rewrite=True, should call LLM to rewrite the query."""
        client = _make_openai_client("Answer")
        # rewrite 应调用 chat.completions.create（第一次用于改写，第二次用于生成回答）
        rewrite_response = MagicMock()
        rewrite_response.choices = [MagicMock()]
        rewrite_response.choices[0].message.content = "Rewritten query text"
        answer_response = MagicMock()
        answer_response.choices = [MagicMock()]
        answer_response.choices[0].message.content = "Final answer"
        client.chat.completions.create.side_effect = [rewrite_response, answer_response]

        engine = RAGEngine(retriever=mock_retriever, openai_client=client)
        response = engine.query("original question", top_k=3, use_rewrite=True)
        assert response.rewritten_query == "Rewritten query text"
        # 检查 retriever 被用改写后的查询调用
        mock_retriever.similarity_search.assert_called_once_with("Rewritten query text", top_k=3)

    def test_query_without_rewrite(self, mock_retriever):
        """When use_rewrite=False, rewritten_query should be empty string."""
        client = _make_openai_client("Answer")
        engine = RAGEngine(retriever=mock_retriever, openai_client=client)
        response = engine.query("question", use_rewrite=False)
        assert response.rewritten_query == ""

    def test_query_confidence_is_float(self, mock_retriever):
        """Confidence should be a float between 0 and 1."""
        client = _make_openai_client("Answer with [1] citation [2].")
        engine = RAGEngine(retriever=mock_retriever, openai_client=client)
        response = engine.query("test", use_rewrite=False)
        assert isinstance(response.confidence, float)
        assert 0.0 <= response.confidence <= 1.0


# ---------------------------------------------------------------------------
# RAGEngine.rewrite_query
# ---------------------------------------------------------------------------


class TestRewriteQuery:
    """Test rewrite_query method."""

    def test_rewrite_returns_string(self):
        """rewrite_query should return the LLM-rewritten query string."""
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Optimized query"
        client.chat.completions.create.return_value = response

        retriever = MagicMock()
        engine = RAGEngine(retriever=retriever, openai_client=client)
        result = engine.rewrite_query("原始問題")
        assert result == "Optimized query"

    def test_rewrite_falls_back_on_error(self):
        """If LLM call fails, rewrite_query should return the original question."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        retriever = MagicMock()
        engine = RAGEngine(retriever=retriever, openai_client=client)
        result = engine.rewrite_query("原始問題")
        assert result == "原始問題"


# ---------------------------------------------------------------------------
# RAGEngine.build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    """Test build_context method."""

    def test_build_context_with_chunks(self):
        """Should format chunks with numbered markers [1], [2], etc."""
        retriever = MagicMock()
        client = MagicMock()
        engine = RAGEngine(retriever=retriever, openai_client=client)

        chunks = [
            {"content": "First chunk content", "metadata": {"source": "doc1.pdf", "page": 1, "title": "Section A"}},
            {"content": "Second chunk content", "metadata": {"source": "doc2.pdf", "page": 2, "title": "Section B"}},
        ]
        context = engine.build_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context
        assert "doc1.pdf" in context

    def test_build_context_empty_chunks(self):
        """Should return a fallback message when no chunks are provided."""
        retriever = MagicMock()
        client = MagicMock()
        engine = RAGEngine(retriever=retriever, openai_client=client)

        context = engine.build_context([])
        assert "没有找到相关上下文" in context or "沒有找到相關上下文" in context

    def test_build_context_includes_source_info(self):
        """Context should include source file, page, and title."""
        retriever = MagicMock()
        client = MagicMock()
        engine = RAGEngine(retriever=retriever, openai_client=client)

        chunks = [
            {"content": "Content", "metadata": {"source": "regulation.pdf", "page": 5, "title": "Article 3"}},
        ]
        context = engine.build_context(chunks)
        assert "regulation.pdf" in context
        assert "5" in context
        assert "Article 3" in context
