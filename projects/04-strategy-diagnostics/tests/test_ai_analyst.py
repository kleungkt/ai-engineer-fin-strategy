"""Tests for the AI analyst module with mocked OpenAI."""

from unittest.mock import MagicMock, patch

import pytest

from src.ai_analyst import generate_analysis, generate_comparison
from src.models import DiagnosticReport, MetricRating


@pytest.fixture
def sample_report() -> DiagnosticReport:
    return DiagnosticReport(
        overall_score=72.5,
        metrics={
            "sharpe_ratio": MetricRating(value=1.6, rating="good", explanation="Good Sharpe"),
            "annual_return": MetricRating(value=0.20, rating="good", explanation="Good return"),
            "max_drawdown": MetricRating(value=0.12, rating="acceptable", explanation="Acceptable DD"),
            "win_rate": MetricRating(value=0.55, rating="good", explanation="Good win rate"),
        },
        strengths=["Sharpe ratio is good"],
        weaknesses=[],
        suggestions=["Consider tighter stops"],
        risk_warnings=[],
    )


@pytest.fixture
def mock_openai_response():
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "## Overall Assessment\nThe strategy demonstrates solid risk-adjusted returns.\n\n"
        "## Risk Profile\nModerate risk with acceptable drawdown levels.\n\n"
        "## Improvement Suggestions\nConsider implementing dynamic position sizing.\n\n"
        "## Market Condition Suitability\nBest suited for trending markets."
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestGenerateAnalysis:
    """Tests for generate_analysis function."""

    @patch("src.ai_analyst.OpenAI")
    def test_returns_string(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_analysis(sample_report)
        assert isinstance(result, str)

    @patch("src.ai_analyst.OpenAI")
    def test_calls_openai(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_analysis(sample_report, strategy_name="TestStrat", params={"window": 20})
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"

    @patch("src.ai_analyst.OpenAI")
    def test_includes_strategy_name(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_analysis(sample_report, strategy_name="MomentumAlpha")
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "MomentumAlpha" in user_msg

    @patch("src.ai_analyst.OpenAI")
    def test_includes_params(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_analysis(sample_report, params={"lookback": 50})
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "lookback" in user_msg

    @patch("src.ai_analyst.OpenAI")
    def test_includes_metrics(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_analysis(sample_report)
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "sharpe_ratio" in user_msg
        assert "1.6" in user_msg

    @patch("src.ai_analyst.OpenAI")
    def test_returns_content(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_analysis(sample_report)
        assert "Assessment" in result

    @patch("src.ai_analyst.OpenAI")
    def test_default_params_none(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        # Should not raise with None params
        result = generate_analysis(sample_report, params=None)
        assert isinstance(result, str)


class TestGenerateComparison:
    """Tests for generate_comparison function."""

    @patch("src.ai_analyst.OpenAI")
    def test_returns_string(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        result = generate_comparison([("Alpha", sample_report)])
        assert isinstance(result, str)

    @patch("src.ai_analyst.OpenAI")
    def test_calls_openai(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_comparison([("Alpha", sample_report), ("Beta", sample_report)])
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.ai_analyst.OpenAI")
    def test_includes_all_names(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_comparison([("Alpha", sample_report), ("Beta", sample_report)])
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Alpha" in user_msg
        assert "Beta" in user_msg

    @patch("src.ai_analyst.OpenAI")
    def test_includes_scores(self, mock_openai_cls, sample_report, mock_openai_response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_cls.return_value = mock_client

        generate_comparison([("Alpha", sample_report)])
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "72.5" in user_msg
