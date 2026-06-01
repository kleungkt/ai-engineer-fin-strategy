"""Tests for the formatter module."""

import json

import pytest

from src.formatter import format_comparison, format_report
from src.models import DiagnosticReport, MetricRating


@pytest.fixture
def sample_report() -> DiagnosticReport:
    """Create a sample report for formatting tests."""
    return DiagnosticReport(
        overall_score=75.0,
        metrics={
            "sharpe_ratio": MetricRating(value=1.8, rating="good", explanation="Good Sharpe ratio"),
            "annual_return": MetricRating(value=0.25, rating="good", explanation="Solid returns"),
            "max_drawdown": MetricRating(value=0.12, rating="acceptable", explanation="Acceptable drawdown"),
            "win_rate": MetricRating(value=0.58, rating="good", explanation="Good win rate"),
        },
        strengths=["Sharpe ratio is good", "Win rate is good"],
        weaknesses=["Drawdown is acceptable"],
        suggestions=["Consider tighter stops"],
        risk_warnings=[],
    )


class TestFormatReportText:
    """Tests for text format."""

    def test_contains_header(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "DIAGNOSTIC REPORT" in output

    def test_contains_score(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "75.0" in output

    def test_contains_metrics(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "sharpe_ratio" in output
        assert "annual_return" in output

    def test_contains_emoji(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "✅" in output or "⚠️" in output

    def test_contains_strengths(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "Strengths" in output

    def test_contains_suggestions(self, sample_report):
        output = format_report(sample_report, format="text")
        assert "Suggestions" in output


class TestFormatReportMarkdown:
    """Tests for markdown format."""

    def test_contains_markdown_header(self, sample_report):
        output = format_report(sample_report, format="markdown")
        assert "# Strategy Diagnostic Report" in output

    def test_contains_table(self, sample_report):
        output = format_report(sample_report, format="markdown")
        assert "| Metric |" in output
        assert "|--------|" in output

    def test_contains_metrics_in_table(self, sample_report):
        output = format_report(sample_report, format="markdown")
        assert "sharpe_ratio" in output
        assert "1.8000" in output

    def test_contains_score(self, sample_report):
        output = format_report(sample_report, format="markdown")
        assert "75.0" in output

    def test_contains_strengths_list(self, sample_report):
        output = format_report(sample_report, format="markdown")
        assert "- ✅" in output


class TestFormatReportJson:
    """Tests for JSON format."""

    def test_valid_json(self, sample_report):
        output = format_report(sample_report, format="json")
        data = json.loads(output)
        assert "overall_score" in data

    def test_json_score(self, sample_report):
        output = format_report(sample_report, format="json")
        data = json.loads(output)
        assert data["overall_score"] == 75.0

    def test_json_metrics(self, sample_report):
        output = format_report(sample_report, format="json")
        data = json.loads(output)
        assert "sharpe_ratio" in data["metrics"]


class TestFormatComparison:
    """Tests for format_comparison."""

    def test_empty_comparison(self):
        output = format_comparison([])
        assert "No strategies" in output

    def test_single_strategy(self, sample_report):
        output = format_comparison([("Alpha", sample_report)])
        assert "Alpha" in output
        assert "COMPARISON" in output

    def test_two_strategies(self, sample_report):
        other = DiagnosticReport(
            overall_score=60.0,
            metrics={
                "sharpe_ratio": MetricRating(value=0.8, rating="acceptable", explanation="OK"),
                "annual_return": MetricRating(value=0.10, rating="acceptable", explanation="OK"),
                "max_drawdown": MetricRating(value=0.20, rating="acceptable", explanation="OK"),
                "win_rate": MetricRating(value=0.45, rating="acceptable", explanation="OK"),
            },
            strengths=[],
            weaknesses=["Low Sharpe"],
            suggestions=["Improve entries"],
            risk_warnings=[],
        )
        output = format_comparison([("Alpha", sample_report), ("Beta", other)])
        assert "Alpha" in output
        assert "Beta" in output
        assert "🏆" in output

    def test_best_strategy_recommended(self, sample_report):
        other = DiagnosticReport(
            overall_score=40.0,
            metrics={
                "sharpe_ratio": MetricRating(value=0.5, rating="acceptable", explanation="OK"),
                "annual_return": MetricRating(value=0.06, rating="acceptable", explanation="OK"),
                "max_drawdown": MetricRating(value=0.20, rating="acceptable", explanation="OK"),
                "win_rate": MetricRating(value=0.42, rating="acceptable", explanation="OK"),
            },
            strengths=[],
            weaknesses=[],
            suggestions=[],
            risk_warnings=[],
        )
        output = format_comparison([("Alpha", sample_report), ("Beta", other)])
        assert "Alpha" in output.split("🏆")[1]  # Alpha should be recommended
