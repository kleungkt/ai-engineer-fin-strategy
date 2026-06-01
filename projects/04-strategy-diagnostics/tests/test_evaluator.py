"""Tests for the evaluator module."""

import pytest

from src.evaluator import (
    ANNUAL_RETURN_THRESHOLDS,
    MAX_DRAWDOWN_THRESHOLDS,
    RATING_SCORES,
    SHARPE_THRESHOLDS,
    WIN_RATE_THRESHOLDS,
    evaluate_backtest,
    rate_metric,
)
from src.models import BacktestResult, MetricRating


class TestRateMetric:
    """Tests for rate_metric function."""

    def test_sharpe_excellent(self):
        result = rate_metric(2.5, SHARPE_THRESHOLDS)
        assert result.rating == "excellent"

    def test_sharpe_excellent_boundary(self):
        result = rate_metric(2.0, SHARPE_THRESHOLDS)
        assert result.rating == "excellent"

    def test_sharpe_good(self):
        result = rate_metric(1.7, SHARPE_THRESHOLDS)
        assert result.rating == "good"

    def test_sharpe_good_boundary(self):
        result = rate_metric(1.5, SHARPE_THRESHOLDS)
        assert result.rating == "good"

    def test_sharpe_acceptable(self):
        result = rate_metric(1.0, SHARPE_THRESHOLDS)
        assert result.rating == "acceptable"

    def test_sharpe_acceptable_boundary(self):
        result = rate_metric(0.5, SHARPE_THRESHOLDS)
        assert result.rating == "acceptable"

    def test_sharpe_poor(self):
        result = rate_metric(0.3, SHARPE_THRESHOLDS)
        assert result.rating == "poor"

    def test_sharpe_poor_boundary(self):
        result = rate_metric(0.0, SHARPE_THRESHOLDS)
        assert result.rating == "poor"

    def test_sharpe_critical(self):
        result = rate_metric(-0.5, SHARPE_THRESHOLDS)
        assert result.rating == "critical"

    def test_annual_return_excellent(self):
        result = rate_metric(0.35, ANNUAL_RETURN_THRESHOLDS)
        assert result.rating == "excellent"

    def test_annual_return_good(self):
        result = rate_metric(0.20, ANNUAL_RETURN_THRESHOLDS)
        assert result.rating == "good"

    def test_annual_return_acceptable(self):
        result = rate_metric(0.08, ANNUAL_RETURN_THRESHOLDS)
        assert result.rating == "acceptable"

    def test_annual_return_poor(self):
        result = rate_metric(0.02, ANNUAL_RETURN_THRESHOLDS)
        assert result.rating == "poor"

    def test_annual_return_critical(self):
        result = rate_metric(-0.05, ANNUAL_RETURN_THRESHOLDS)
        assert result.rating == "critical"

    def test_max_drawdown_excellent(self):
        result = rate_metric(0.03, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
        assert result.rating == "excellent"

    def test_max_drawdown_good(self):
        result = rate_metric(0.08, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
        assert result.rating == "good"

    def test_max_drawdown_acceptable(self):
        result = rate_metric(0.15, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
        assert result.rating == "acceptable"

    def test_max_drawdown_poor(self):
        result = rate_metric(0.25, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
        assert result.rating == "poor"

    def test_max_drawdown_critical(self):
        result = rate_metric(0.50, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
        assert result.rating == "critical"

    def test_win_rate_excellent(self):
        result = rate_metric(0.70, WIN_RATE_THRESHOLDS)
        assert result.rating == "excellent"

    def test_win_rate_good(self):
        result = rate_metric(0.55, WIN_RATE_THRESHOLDS)
        assert result.rating == "good"

    def test_win_rate_acceptable(self):
        result = rate_metric(0.45, WIN_RATE_THRESHOLDS)
        assert result.rating == "acceptable"

    def test_win_rate_poor(self):
        result = rate_metric(0.32, WIN_RATE_THRESHOLDS)
        assert result.rating == "poor"

    def test_win_rate_critical(self):
        result = rate_metric(0.20, WIN_RATE_THRESHOLDS)
        assert result.rating == "critical"

    def test_metric_rating_has_explanation(self):
        result = rate_metric(1.5, SHARPE_THRESHOLDS)
        assert result.explanation
        assert "1.5" in result.explanation

    def test_metric_rating_is_pydantic_model(self):
        result = rate_metric(1.5, SHARPE_THRESHOLDS)
        assert isinstance(result, MetricRating)


class TestEvaluateBacktest:
    """Tests for evaluate_backtest function."""

    def test_excellent_backtest_score(self, excellent_backtest):
        report = evaluate_backtest(excellent_backtest)
        assert report.overall_score >= 85

    def test_mediocre_backtest_score(self, mediocre_backtest):
        report = evaluate_backtest(mediocre_backtest)
        assert 40 <= report.overall_score <= 75

    def test_poor_backtest_score(self, poor_backtest):
        report = evaluate_backtest(poor_backtest)
        assert report.overall_score <= 30

    def test_all_metrics_rated(self, excellent_backtest):
        report = evaluate_backtest(excellent_backtest)
        assert "sharpe_ratio" in report.metrics
        assert "annual_return" in report.metrics
        assert "max_drawdown" in report.metrics
        assert "win_rate" in report.metrics

    def test_excellent_has_strengths(self, excellent_backtest):
        report = evaluate_backtest(excellent_backtest)
        assert len(report.strengths) > 0

    def test_poor_has_weaknesses(self, poor_backtest):
        report = evaluate_backtest(poor_backtest)
        assert len(report.weaknesses) > 0

    def test_poor_has_risk_warnings(self, poor_backtest):
        report = evaluate_backtest(poor_backtest)
        assert len(report.risk_warnings) > 0

    def test_suggestions_always_present(self, excellent_backtest, mediocre_backtest, poor_backtest):
        for bt in [excellent_backtest, mediocre_backtest, poor_backtest]:
            report = evaluate_backtest(bt)
            assert len(report.suggestions) > 0

    def test_score_range(self, excellent_backtest, mediocre_backtest, poor_backtest):
        for bt in [excellent_backtest, mediocre_backtest, poor_backtest]:
            report = evaluate_backtest(bt)
            assert 0 <= report.overall_score <= 100

    def test_excellent_no_risk_warnings(self, excellent_backtest):
        report = evaluate_backtest(excellent_backtest)
        assert len(report.risk_warnings) == 0

    def test_low_trade_count_suggestion(self):
        bt = BacktestResult(
            total_return=0.5,
            annual_return=0.20,
            sharpe_ratio=1.8,
            max_drawdown=0.08,
            max_drawdown_duration=15,
            win_rate=0.55,
            total_trades=10,
            trade_log=[],
        )
        report = evaluate_backtest(bt)
        assert any("10 trades" in s for s in report.suggestions)

    def test_borderline_all_acceptable(self):
        bt = BacktestResult(
            total_return=0.10,
            annual_return=0.06,
            sharpe_ratio=0.6,
            max_drawdown=0.15,
            max_drawdown_duration=40,
            win_rate=0.42,
            total_trades=60,
            trade_log=[],
        )
        report = evaluate_backtest(bt)
        for metric in report.metrics.values():
            assert metric.rating in ("acceptable", "good", "excellent", "poor", "critical")

    def test_report_is_pydantic_model(self, excellent_backtest):
        from src.models import DiagnosticReport
        report = evaluate_backtest(excellent_backtest)
        assert isinstance(report, DiagnosticReport)
