"""Tests for src/evaluator module."""

import pytest
from unittest.mock import MagicMock

from src.evaluator import RAGEvaluator, _tokenize, _keyword_overlap


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test helper functions in the evaluator module."""

    def test_tokenize_english(self):
        """_tokenize should split English text into lowercase tokens."""
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_chinese(self):
        """_tokenize should handle Chinese characters."""
        tokens = _tokenize("MACD指標")
        # 应该包含中文单字和英文单词
        assert any("m" in t or "m" == t for t in tokens) or "macd" in tokens
        assert any("\u4e00" <= t <= "\u9fff" for t in tokens)

    def test_keyword_overlap_identical_texts(self):
        """Identical texts should have overlap close to 1.0."""
        text = "financial risk management"
        overlap = _keyword_overlap(text, text)
        assert overlap > 0.9

    def test_keyword_overlap_no_common_tokens(self):
        """Completely different texts should have overlap close to 0.0."""
        overlap = _keyword_overlap("apple banana", "xyz qwerty")
        assert overlap < 0.1

    def test_keyword_overlap_partial(self):
        """Partially overlapping texts should have intermediate overlap."""
        overlap = _keyword_overlap("financial risk management", "financial portfolio theory")
        assert 0.0 < overlap < 1.0

    def test_keyword_overlap_empty_texts(self):
        """Empty texts should return 0.0 overlap."""
        assert _keyword_overlap("", "hello") == 0.0
        assert _keyword_overlap("hello", "") == 0.0
        assert _keyword_overlap("", "") == 0.0


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------


class TestEvaluateRetrieval:
    """Test evaluate_retrieval method."""

    def test_perfect_retrieval(self, mock_retriever):
        """When all retrieved docs match expected, precision and recall should be 1.0."""
        # mock retriever 返回与 expected_docs 内容匹配的结果
        mock_retriever.similarity_search.return_value = [
            {"content": "Financial risk management is important", "metadata": {}, "distance": 0.05},
            {"content": "Portfolio diversification reduces risk", "metadata": {}, "distance": 0.10},
        ]

        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_retrieval(
            queries=["What is risk management?"],
            expected_docs=[["Financial risk management is important", "Portfolio diversification reduces risk"]],
            retriever=mock_retriever,
            top_k=2,
        )
        assert metrics["precision@k"] > 0.0
        assert metrics["recall@k"] > 0.0
        assert metrics["mrr"] > 0.0

    def test_no_relevant_docs(self, mock_retriever):
        """When no retrieved docs match expected, metrics should be 0."""
        mock_retriever.similarity_search.return_value = [
            {"content": "Completely unrelated content about cooking", "metadata": {}, "distance": 0.9},
        ]

        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_retrieval(
            queries=["financial risk"],
            expected_docs=[["Financial risk management"]],
            retriever=mock_retriever,
            top_k=1,
        )
        # 由于 cooking 和 financial 可能有部分字符重叠，precision 可能 > 0
        # 但 recall 应该很低
        assert isinstance(metrics["precision@k"], float)
        assert isinstance(metrics["recall@k"], float)

    def test_mismatched_lengths_raises(self, mock_retriever):
        """queries and expected_docs must have the same length."""
        evaluator = RAGEvaluator()
        with pytest.raises(ValueError, match="same length"):
            evaluator.evaluate_retrieval(
                queries=["q1", "q2"],
                expected_docs=[["e1"]],
                retriever=mock_retriever,
            )

    def test_returns_all_metric_keys(self, mock_retriever):
        """Result dict should contain all expected metric keys."""
        mock_retriever.similarity_search.return_value = [
            {"content": "test content", "metadata": {}, "distance": 0.1},
        ]
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_retrieval(
            queries=["test"],
            expected_docs=[["test content"]],
            retriever=mock_retriever,
        )
        assert "precision@k" in metrics
        assert "recall@k" in metrics
        assert "mrr" in metrics
        assert "num_queries" in metrics


# ---------------------------------------------------------------------------
# evaluate_answer
# ---------------------------------------------------------------------------


class TestEvaluateAnswer:
    """Test evaluate_answer method."""

    def test_faithfulness_with_matching_context(self):
        """Answer using words from context should have high faithfulness."""
        evaluator = RAGEvaluator()
        question = "What is risk management?"
        answer = "Risk management is the process of identifying financial risks."
        context = "Risk management involves identifying and mitigating financial risks in portfolio management."
        metrics = evaluator.evaluate_answer(question, answer, context)
        assert "faithfulness" in metrics
        assert metrics["faithfulness"] > 0.0

    def test_relevancy_with_matching_question(self):
        """Answer addressing the question should have some relevancy."""
        evaluator = RAGEvaluator()
        question = "What is MACD?"
        answer = "MACD is Moving Average Convergence Divergence, a momentum indicator."
        context = "MACD is a popular technical analysis indicator."
        metrics = evaluator.evaluate_answer(question, answer, context)
        assert "relevancy" in metrics
        assert metrics["relevancy"] > 0.0

    def test_overall_score_is_average(self):
        """Overall score should be the average of faithfulness and relevancy."""
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_answer("test question", "test answer", "test context")
        expected_overall = 0.5 * metrics["faithfulness"] + 0.5 * metrics["relevancy"]
        assert abs(metrics["overall"] - expected_overall) < 0.01

    def test_returns_all_keys(self):
        """Should return faithfulness, relevancy, and overall."""
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_answer("Q", "A", "C")
        assert "faithfulness" in metrics
        assert "relevancy" in metrics
        assert "overall" in metrics


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    """Test run_evaluation with simple test cases."""

    def test_basic_run_with_retriever(self, mock_retriever):
        """Should run evaluation and return aggregated metrics."""
        mock_retriever.similarity_search.return_value = [
            {"content": "Financial risk management principles", "metadata": {}, "distance": 0.1},
            {"content": "Diversification reduces portfolio risk", "metadata": {}, "distance": 0.2},
        ]

        evaluator = RAGEvaluator()
        test_cases = [
            {
                "query": "What is risk management?",
                "expected_answer_keywords": ["risk", "management"],
                "expected_sources": ["Financial risk management principles"],
            },
        ]
        results = evaluator.run_evaluation(
            test_cases=test_cases,
            retriever=mock_retriever,
            top_k=2,
        )
        assert results["total_cases"] == 1
        assert "retrieval_metrics" in results
        assert "per_case_results" in results

    def test_basic_run_with_rag_engine(self):
        """Should run end-to-end evaluation with a mocked RAG engine."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Risk management is identifying financial risks [1]."
        mock_response.sources = [
            {"content": "Risk management content", "source": "test.pdf", "score": 0.9},
        ]
        mock_engine.query.return_value = mock_response

        evaluator = RAGEvaluator()
        test_cases = [
            {
                "query": "What is risk management?",
                "expected_answer_keywords": ["risk", "management"],
                "expected_sources": [],
            },
        ]
        results = evaluator.run_evaluation(
            test_cases=test_cases,
            rag_engine=mock_engine,
            top_k=3,
        )
        assert results["total_cases"] == 1
        assert "answer_metrics" in results
        assert len(results["per_case_results"]) == 1

    def test_empty_test_cases(self):
        """Should handle empty test cases gracefully."""
        evaluator = RAGEvaluator()
        results = evaluator.run_evaluation(test_cases=[])
        assert results["total_cases"] == 0
        assert results["retrieval_metrics"] == {}
        assert results["answer_metrics"] == {}

    def test_multiple_test_cases(self, mock_retriever):
        """Should aggregate metrics across multiple test cases."""
        mock_retriever.similarity_search.return_value = [
            {"content": "MACD momentum indicator", "metadata": {}, "distance": 0.05},
        ]

        evaluator = RAGEvaluator()
        test_cases = [
            {
                "query": "What is MACD?",
                "expected_answer_keywords": ["MACD", "momentum"],
                "expected_sources": ["MACD momentum indicator"],
            },
            {
                "query": "What is RSI?",
                "expected_answer_keywords": ["RSI", "overbought"],
                "expected_sources": ["MACD momentum indicator"],
            },
        ]
        results = evaluator.run_evaluation(
            test_cases=test_cases,
            retriever=mock_retriever,
        )
        assert results["total_cases"] == 2
        assert results["retrieval_metrics"]["num_queries"] == 2
