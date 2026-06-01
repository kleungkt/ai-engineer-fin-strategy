"""Tests for the evaluator module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import (
    EvalResult,
    evaluate_generation,
    evaluate_qa,
    evaluate_sentiment,
    run_benchmark,
    _simple_bleu,
    _simple_rouge_l,
)


# ---------------------------------------------------------------------------
# EvalResult model
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_creation(self):
        r = EvalResult(metric_name="accuracy", score=0.85, details={"correct": 85, "total": 100})
        assert r.metric_name == "accuracy"
        assert r.score == 0.85
        assert r.details["correct"] == 85

    def test_default_details(self):
        r = EvalResult(metric_name="test", score=1.0)
        assert r.details == {}

    def test_model_dump(self):
        r = EvalResult(metric_name="f1", score=0.9)
        d = r.model_dump()
        assert d["metric_name"] == "f1"
        assert d["score"] == 0.9


# ---------------------------------------------------------------------------
# BLEU / ROUGE helpers
# ---------------------------------------------------------------------------

class TestSimpleBleu:
    def test_perfect_match(self):
        score = _simple_bleu("hello world", "hello world")
        assert score == 1.0

    def test_no_overlap(self):
        score = _simple_bleu("hello world", "foo bar")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _simple_bleu("the quick brown fox", "the slow brown dog")
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        score = _simple_bleu("hello world", "")
        assert score == 0.0

    def test_empty_reference(self):
        score = _simple_bleu("", "hello world")
        assert score == 0.0


class TestSimpleRougeL:
    def test_perfect_match(self):
        score = _simple_rouge_l("hello world", "hello world")
        assert score == 1.0

    def test_no_overlap(self):
        score = _simple_rouge_l("aaa bbb", "ccc ddd")
        assert score == 0.0

    def test_partial_lcs(self):
        score = _simple_rouge_l("a b c d", "a x c y")
        assert 0.0 < score < 1.0

    def test_empty_inputs(self):
        assert _simple_rouge_l("", "hello") == 0.0
        assert _simple_rouge_l("hello", "") == 0.0


# ---------------------------------------------------------------------------
# evaluate_sentiment (mocked)
# ---------------------------------------------------------------------------

class TestEvaluateSentiment:
    @patch("src.evaluator._load_model")
    @patch("src.evaluator._generate_text")
    def test_perfect_accuracy(self, mock_generate, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.side_effect = ["bullish", "bearish"]

        test_data = [
            {"input": "Stock soars!", "output": "The sentiment is bullish.", "category": "sentiment"},
            {"input": "Market crashes!", "output": "The sentiment is bearish.", "category": "sentiment"},
        ]
        result = evaluate_sentiment("/fake/model", test_data)
        assert result.metric_name == "sentiment_accuracy"
        assert result.score == 1.0

    @patch("src.evaluator._load_model")
    @patch("src.evaluator._generate_text")
    def test_partial_accuracy(self, mock_generate, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.side_effect = ["bullish", "bullish"]  # second is wrong

        test_data = [
            {"input": "Stock soars!", "output": "The sentiment is bullish.", "category": "sentiment"},
            {"input": "Market crashes!", "output": "The sentiment is bearish.", "category": "sentiment"},
        ]
        result = evaluate_sentiment("/fake/model", test_data)
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# evaluate_qa (mocked)
# ---------------------------------------------------------------------------

class TestEvaluateQa:
    @patch("src.evaluator._load_model")
    @patch("src.evaluator._generate_text")
    def test_returns_eval_result(self, mock_generate, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.return_value = "P/E ratio is price to earnings"

        test_data = [
            {"input": "What is P/E?", "output": "P/E ratio is price to earnings ratio", "category": "qa"},
        ]
        result = evaluate_qa("/fake/model", test_data)
        assert result.metric_name == "qa_quality"
        assert "bleu" in result.details
        assert "rouge_l" in result.details
        assert result.score >= 0.0

    @patch("src.evaluator._load_model")
    @patch("src.evaluator._generate_text")
    def test_perfect_qa_score(self, mock_generate, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.return_value = "compound interest is calculated on principal and accumulated interest"

        test_data = [
            {"input": "What is compound interest?", "output": "compound interest is calculated on principal and accumulated interest", "category": "qa"},
        ]
        result = evaluate_qa("/fake/model", test_data)
        assert result.score > 0.9


# ---------------------------------------------------------------------------
# evaluate_generation (mocked)
# ---------------------------------------------------------------------------

class TestEvaluateGeneration:
    @patch("src.evaluator._load_model")
    @patch("src.evaluator._generate_text")
    def test_returns_list(self, mock_generate, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.side_effect = ["output1", "output2", "output3"]

        results = evaluate_generation("/fake/model", ["p1", "p2", "p3"])
        assert len(results) == 3
        assert results[0] == "output1"


# ---------------------------------------------------------------------------
# run_benchmark (mocked)
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    @patch("src.evaluator.evaluate_sentiment")
    @patch("src.evaluator.evaluate_qa")
    @patch("src.evaluator.evaluate_generation")
    def test_benchmark_by_category(self, mock_gen, mock_qa, mock_sent):
        mock_sent.return_value = EvalResult(metric_name="sentiment_accuracy", score=0.9)
        mock_qa.return_value = EvalResult(metric_name="qa_quality", score=0.7)
        mock_gen.return_value = ["gen1"]

        test_data = [
            {"input": "news1", "output": "bullish", "category": "sentiment"},
            {"input": "news2", "output": "bearish", "category": "sentiment"},
            {"input": "What is X?", "output": "X is ...", "category": "qa"},
        ]
        results = run_benchmark("/fake/model", test_data)
        assert "sentiment_accuracy" in results
        assert "qa_quality" in results
        assert "generation_count" in results

    @patch("src.evaluator.evaluate_generation")
    def test_benchmark_empty_data(self, mock_gen):
        mock_gen.return_value = []
        results = run_benchmark("/fake/model", [])
        # No sentiment or qa data, so only generation_count may be present
        assert isinstance(results, dict)
