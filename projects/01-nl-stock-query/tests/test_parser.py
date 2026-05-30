"""Tests for the LLM-based query parser (parser.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from parser import (
    IndicatorCondition,
    IntentType,
    QueryIntent,
    get_extraction_prompt,
    parse_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_openai_response():
    """Return a factory that builds a mock OpenAI chat completion response."""

    def _make(arguments: dict | str):
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)

        tool_call = MagicMock()
        tool_call.function.arguments = arguments

        message = MagicMock()
        message.tool_calls = [tool_call]
        message.content = None

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "tool_calls"

        response = MagicMock()
        response.choices = [choice]
        return response

    return _make


# ---------------------------------------------------------------------------
# Happy-path: parse_query returns a valid QueryIntent
# ---------------------------------------------------------------------------

class TestParseQueryHappyPath:
    """parse_query with a mocked OpenAI response should return correct QueryIntent."""

    @patch("parser.OpenAI")
    def test_single_indicator(self, mock_openai_cls, _mock_openai_response):
        """Single RSI condition."""
        llm_args = {
            "intent_type": "indicator_screener",
            "indicators": [
                {"name": "RSI", "comparison": "below", "value": 30, "params": {}}
            ],
            "stock_scope": "A股",
            "time_range": 30,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        result = parse_query("找出 RSI < 30 的 A 股")

        assert isinstance(result, QueryIntent)
        assert result.intent_type == IntentType.INDICATOR_SCREENER
        assert len(result.indicators) == 1
        assert result.indicators[0].name == "RSI"
        assert result.indicators[0].comparison == "below"
        assert result.indicators[0].value == 30
        assert result.stock_scope == "A股"
        assert result.time_range == 30

    @patch("parser.OpenAI")
    def test_multiple_indicators(self, mock_openai_cls, _mock_openai_response):
        """Multiple indicator conditions."""
        llm_args = {
            "intent_type": "indicator_screener",
            "indicators": [
                {"name": "MACD", "comparison": "crossover", "value": None, "params": {}},
                {"name": "RSI", "comparison": "below", "value": 40, "params": {"period": 14}},
            ],
            "stock_scope": "美股",
            "time_range": 60,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        result = parse_query("美股 MACD 黄金交叉且 RSI < 40")

        assert result.intent_type == IntentType.INDICATOR_SCREENER
        assert len(result.indicators) == 2
        assert result.indicators[0].name == "MACD"
        assert result.indicators[0].comparison == "crossover"
        assert result.indicators[1].name == "RSI"
        assert result.indicators[1].params == {"period": 14}
        assert result.stock_scope == "美股"
        assert result.time_range == 60

    @patch("parser.OpenAI")
    def test_pattern_query_intent(self, mock_openai_cls, _mock_openai_response):
        """Non-screener intent type."""
        llm_args = {
            "intent_type": "pattern_query",
            "indicators": [],
            "stock_scope": "A股",
            "time_range": 30,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        result = parse_query("出现头肩底的股票")

        assert result.intent_type == IntentType.PATTERN_QUERY
        assert result.indicators == []


# ---------------------------------------------------------------------------
# Validation: bad data should raise
# ---------------------------------------------------------------------------

class TestParseQueryValidation:
    """Validation edge-cases for the parser."""

    def test_empty_input_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_query("")

    def test_whitespace_only_raises(self):
        """Whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_query("   ")

    @patch("parser.OpenAI")
    def test_invalid_comparison_raises(self, mock_openai_cls, _mock_openai_response):
        """Invalid comparison value should fail Pydantic validation."""
        llm_args = {
            "intent_type": "indicator_screener",
            "indicators": [
                {"name": "RSI", "comparison": "INVALID_CMP", "value": 50, "params": {}}
            ],
            "stock_scope": "A股",
            "time_range": 30,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError, match="validation"):
            parse_query("RSI something")

    @patch("parser.OpenAI")
    def test_invalid_intent_type_raises(self, mock_openai_cls, _mock_openai_response):
        """Unknown intent_type should fail validation."""
        llm_args = {
            "intent_type": "totally_invalid",
            "indicators": [],
            "stock_scope": "A股",
            "time_range": 30,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError):
            parse_query("random query")

    @patch("parser.OpenAI")
    def test_no_tool_calls_raises(self, mock_openai_cls):
        """LLM returning no tool_calls should raise ValueError."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "I don't understand"
        response.choices[0].finish_reason = "stop"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError, match="did not return a tool call"):
            parse_query("something ambiguous")

    @patch("parser.OpenAI")
    def test_invalid_json_raises(self, mock_openai_cls):
        """Malformed JSON in tool call arguments should raise ValueError."""
        tool_call = MagicMock()
        tool_call.function.arguments = "NOT-VALID-JSON{{}}"

        message = MagicMock()
        message.tool_calls = [tool_call]
        message.content = None

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "tool_calls"

        response = MagicMock()
        response.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError, match="JSON"):
            parse_query("some query")


# ---------------------------------------------------------------------------
# Edge-cases
# ---------------------------------------------------------------------------

class TestParseQueryEdgeCases:
    """Edge-case scenarios."""

    @patch("parser.OpenAI")
    def test_openai_api_failure_raises_runtime_error(self, mock_openai_cls):
        """Network / API error should surface as RuntimeError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("timeout")
        mock_openai_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            parse_query("RSI < 30")

    @patch("parser.OpenAI")
    def test_time_range_boundary_values(self, mock_openai_cls, _mock_openai_response):
        """time_range=1 and time_range=365 are both valid."""
        for tr in (1, 365):
            llm_args = {
                "intent_type": "indicator_screener",
                "indicators": [],
                "stock_scope": "A股",
                "time_range": tr,
            }
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
            mock_openai_cls.return_value = mock_client

            result = parse_query("test query")
            assert result.time_range == tr

    @patch("parser.OpenAI")
    def test_time_range_out_of_bounds_raises(self, mock_openai_cls, _mock_openai_response):
        """time_range=0 or 366 should fail Pydantic validation (ge=1, le=365)."""
        for tr in (0, 366):
            llm_args = {
                "intent_type": "indicator_screener",
                "indicators": [],
                "stock_scope": "A股",
                "time_range": tr,
            }
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
            mock_openai_cls.return_value = mock_client

            with pytest.raises(ValueError):
                parse_query("test query")

    @patch("parser.OpenAI")
    def test_model_kwarg_passed_through(self, mock_openai_cls, _mock_openai_response):
        """Custom model name should be forwarded to OpenAI."""
        llm_args = {
            "intent_type": "indicator_screener",
            "indicators": [],
            "stock_scope": "A股",
            "time_range": 30,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(llm_args)
        mock_openai_cls.return_value = mock_client

        parse_query("test", model="gpt-4o")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs[1].get("model") == "gpt-4o"


# ---------------------------------------------------------------------------
# IndicatorCondition model tests
# ---------------------------------------------------------------------------

class TestIndicatorCondition:
    """Direct model validation for IndicatorCondition."""

    def test_valid_conditions(self):
        for cmp in ("above", "below", "crossover", "crossunder"):
            cond = IndicatorCondition(name="RSI", comparison=cmp)
            assert cond.comparison == cmp

    def test_comparison_case_insensitive(self):
        cond = IndicatorCondition(name="MACD", comparison="ABOVE")
        assert cond.comparison == "above"

    def test_invalid_comparison_raises(self):
        with pytest.raises(Exception):
            IndicatorCondition(name="RSI", comparison="greater_than")

    def test_default_params_empty(self):
        cond = IndicatorCondition(name="RSI", comparison="below")
        assert cond.params == {}

    def test_default_value_none(self):
        cond = IndicatorCondition(name="RSI", comparison="crossover")
        assert cond.value is None


# ---------------------------------------------------------------------------
# get_extraction_prompt
# ---------------------------------------------------------------------------

class TestGetExtractionPrompt:
    def test_returns_nonempty_string(self):
        prompt = get_extraction_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_key_terms(self):
        prompt = get_extraction_prompt()
        assert "intent_type" in prompt
        assert "MACD" in prompt
