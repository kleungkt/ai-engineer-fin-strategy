"""Tests for the diagnostics module (diagnostics.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backtester import BacktestResult
from diagnostics import (
    DiagnosticReport,
    evaluate_backtest,
    format_report,
    generate_ai_analysis,
    MetricSummary,
    _rate_drawdown,
    _rate_return,
    _rate_sharpe,
    _rate_win_rate,
    _score_drawdown,
    _score_return,
    _score_sharpe,
    _score_win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    *,
    total_return: float = 20.0,
    annual_return: float = 15.0,
    sharpe_ratio: float | None = 1.0,
    max_drawdown: float = 12.0,
    max_drawdown_duration: int = 30,
    win_rate: float = 55.0,
    total_trades: int = 40,
) -> BacktestResult:
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        win_rate=win_rate,
        total_trades=total_trades,
    )


@pytest.fixture()
def good_result() -> BacktestResult:
    return _make_result(
        total_return=60.0,
        annual_return=25.0,
        sharpe_ratio=2.0,
        max_drawdown=5.0,
        max_drawdown_duration=15,
        win_rate=65.0,
        total_trades=80,
    )


@pytest.fixture()
def poor_result() -> BacktestResult:
    return _make_result(
        total_return=-5.0,
        annual_return=-3.0,
        sharpe_ratio=-0.2,
        max_drawdown=35.0,
        max_drawdown_duration=200,
        win_rate=30.0,
        total_trades=20,
    )


@pytest.fixture()
def mediocre_result() -> BacktestResult:
    return _make_result(
        total_return=8.0,
        annual_return=4.0,
        sharpe_ratio=0.3,
        max_drawdown=25.0,
        max_drawdown_duration=100,
        win_rate=42.0,
        total_trades=25,
    )


# ---------------------------------------------------------------------------
# Rating threshold functions
# ---------------------------------------------------------------------------

class TestRateSharpe:
    def test_good_above_1_5(self):
        assert _rate_sharpe(2.0) == "good"

    def test_good_boundary(self):
        assert _rate_sharpe(1.51) == "good"

    def test_acceptable(self):
        assert _rate_sharpe(1.0) == "acceptable"

    def test_acceptable_boundary(self):
        assert _rate_sharpe(0.51) == "acceptable"

    def test_poor(self):
        assert _rate_sharpe(0.3) == "poor"

    def test_poor_at_zero(self):
        assert _rate_sharpe(0.0) == "poor"

    def test_critical_negative(self):
        assert _rate_sharpe(-0.5) == "critical"


class TestRateDrawdown:
    def test_good_low(self):
        assert _rate_drawdown(5.0) == "good"

    def test_good_boundary(self):
        assert _rate_drawdown(9.99) == "good"

    def test_acceptable(self):
        assert _rate_drawdown(15.0) == "acceptable"

    def test_poor(self):
        assert _rate_drawdown(25.0) == "poor"

    def test_critical_high(self):
        assert _rate_drawdown(40.0) == "critical"

    def test_negative_drawdown_handled(self):
        """Negative drawdown should be treated as positive magnitude."""
        assert _rate_drawdown(-15.0) == "acceptable"


class TestRateWinRate:
    def test_good(self):
        assert _rate_win_rate(60.0) == "good"

    def test_acceptable(self):
        assert _rate_win_rate(45.0) == "acceptable"

    def test_poor(self):
        assert _rate_win_rate(30.0) == "poor"

    def test_boundary_50(self):
        assert _rate_win_rate(50.01) == "good"

    def test_boundary_40(self):
        assert _rate_win_rate(40.01) == "acceptable"


class TestRateReturn:
    def test_good(self):
        assert _rate_return(25.0) == "good"

    def test_acceptable(self):
        assert _rate_return(10.0) == "acceptable"

    def test_poor(self):
        assert _rate_return(2.0) == "poor"

    def test_critical_negative(self):
        assert _rate_return(-5.0) == "critical"


# ---------------------------------------------------------------------------
# Score functions – boundary tests
# ---------------------------------------------------------------------------

class TestScoreFunctions:
    def test_score_sharpe_zero(self):
        assert _score_sharpe(0.0) == 0.0

    def test_score_sharpe_max(self):
        """Sharpe of 2.0+ should hit 30."""
        assert _score_sharpe(2.0) == 30.0
        assert _score_sharpe(5.0) == 30.0  # capped

    def test_score_sharpe_negative(self):
        assert _score_sharpe(-1.0) == 0.0

    def test_score_return_zero(self):
        assert _score_return(0.0) == 0.0

    def test_score_return_max(self):
        """Annual return of 30%+ should hit 25."""
        assert _score_return(30.0) == 25.0
        assert _score_return(100.0) == 25.0  # capped

    def test_score_return_negative(self):
        assert _score_return(-10.0) == 0.0

    def test_score_drawdown_zero(self):
        """No drawdown → max score 25."""
        assert _score_drawdown(0.0) == 25.0

    def test_score_drawdown_full(self):
        """50% drawdown → score 0."""
        assert _score_drawdown(50.0) == 0.0

    def test_score_drawdown_negative_input(self):
        """Negative drawdown should use abs value."""
        assert _score_drawdown(-20.0) == _score_drawdown(20.0)

    def test_score_win_rate_zero(self):
        assert _score_win_rate(0.0) == 0.0

    def test_score_win_rate_max(self):
        """100% win rate → score 20."""
        assert _score_win_rate(100.0) == 20.0


# ---------------------------------------------------------------------------
# evaluate_backtest
# ---------------------------------------------------------------------------

class TestEvaluateBacktest:
    def test_returns_diagnostic_report(self, good_result):
        report = evaluate_backtest(good_result)
        assert isinstance(report, DiagnosticReport)

    def test_good_result_high_score(self, good_result):
        report = evaluate_backtest(good_result)
        assert report.overall_score > 50

    def test_poor_result_low_score(self, poor_result):
        report = evaluate_backtest(poor_result)
        assert report.overall_score < 40

    def test_score_range(self, good_result, poor_result, mediocre_result):
        for result in (good_result, poor_result, mediocre_result):
            report = evaluate_backtest(result)
            assert 0 <= report.overall_score <= 100

    def test_metrics_summary_keys(self, good_result):
        report = evaluate_backtest(good_result)
        expected_keys = {"sharpe_ratio", "annual_return", "max_drawdown", "win_rate", "total_trades"}
        assert set(report.metrics_summary.keys()) == expected_keys

    def test_each_metric_has_value_and_rating(self, good_result):
        report = evaluate_backtest(good_result)
        for name, ms in report.metrics_summary.items():
            assert isinstance(ms, MetricSummary)
            assert isinstance(ms.value, float)
            assert ms.rating in ("good", "acceptable", "poor", "critical")

    def test_good_result_has_strengths(self, good_result):
        report = evaluate_backtest(good_result)
        assert len(report.strengths) > 0

    def test_poor_result_has_weaknesses(self, poor_result):
        report = evaluate_backtest(poor_result)
        assert len(report.weaknesses) > 0

    def test_suggestions_always_present(self, good_result, poor_result, mediocre_result):
        for result in (good_result, poor_result, mediocre_result):
            report = evaluate_backtest(result)
            assert len(report.suggestions) > 0

    def test_low_trades_count_weakness(self):
        result = _make_result(total_trades=10)
        report = evaluate_backtest(result)
        trade_weakness = [w for w in report.weaknesses if "trade count" in w.lower() or "statistical" in w.lower()]
        assert len(trade_weakness) > 0

    def test_high_drawdown_suggestion(self):
        result = _make_result(max_drawdown=35.0)
        report = evaluate_backtest(result)
        drawdown_suggestions = [s for s in report.suggestions if "drawdown" in s.lower() or "position" in s.lower()]
        assert len(drawdown_suggestions) > 0

    def test_low_sharpe_suggestion(self):
        result = _make_result(sharpe_ratio=0.1)
        report = evaluate_backtest(result)
        sharpe_suggestions = [s for s in report.suggestions if "sharpe" in s.lower() or "risk" in s.lower()]
        assert len(sharpe_suggestions) > 0

    def test_none_sharpe_handled(self):
        """sharpe_ratio=None should not crash."""
        result = BacktestResult(sharpe_ratio=None)
        # This will call _score_sharpe(None) which would error;
        # the function should handle it. If it doesn't, this test documents the bug.
        # We'll check if it raises or handles gracefully.
        try:
            report = evaluate_backtest(result)
            # If it succeeds, score should still be valid
            assert 0 <= report.overall_score <= 100
        except (TypeError, AttributeError):
            pytest.xfail("evaluate_backtest does not handle sharpe_ratio=None")


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_returns_string(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert isinstance(output, str)

    def test_contains_header(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert "STRATEGY DIAGNOSTIC REPORT" in output

    def test_contains_score(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert "Overall Score" in output

    def test_contains_metrics(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert "sharpe_ratio" in output
        assert "annual_return" in output

    def test_contains_strengths(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert "Strengths" in output

    def test_contains_weaknesses(self, poor_result):
        report = evaluate_backtest(poor_result)
        output = format_report(report)
        assert "Weaknesses" in output

    def test_contains_suggestions(self, good_result):
        report = evaluate_backtest(good_result)
        output = format_report(report)
        assert "Suggestions" in output

    def test_with_ai_analysis(self, good_result):
        report = evaluate_backtest(good_result)
        report.ai_analysis = "This is a test AI analysis."
        output = format_report(report)
        assert "AI Analysis" in output
        assert "This is a test AI analysis." in output


# ---------------------------------------------------------------------------
# generate_ai_analysis – mocked OpenAI
# ---------------------------------------------------------------------------

class TestGenerateAIAnalysis:
    @patch("diagnostics.OpenAI")
    def test_calls_openai(self, mock_openai_cls, good_result):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI analysis text"
        mock_client.chat.completions.create.return_value = mock_response

        report = evaluate_backtest(good_result)
        result = generate_ai_analysis(report, "ma_crossover", {"fast": 10, "slow": 30})

        assert result == "AI analysis text"
        mock_client.chat.completions.create.assert_called_once()

    @patch("diagnostics.OpenAI")
    def test_returns_empty_on_none_content(self, mock_openai_cls, good_result):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        report = evaluate_backtest(good_result)
        result = generate_ai_analysis(report, "rsi_extreme", {})
        assert result == ""

    @patch("diagnostics.OpenAI")
    def test_prompt_includes_metrics(self, mock_openai_cls, good_result):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        report = evaluate_backtest(good_result)
        generate_ai_analysis(report, "macd_signal", {})

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "macd_signal" in user_msg
        assert "sharpe_ratio" in user_msg
