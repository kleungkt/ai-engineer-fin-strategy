"""Tests for the strategy_agent module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from strategy_agent import StrategyAgent, StrategySpec, STRATEGY_SPEC_FUNCTION, CODE_GEN_SYSTEM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Create a StrategyAgent with a mocked OpenAI client."""
    with patch("strategy_agent.OpenAI") as MockOpenAI:
        agent = StrategyAgent(model="gpt-4o", api_key="test-key")
        yield agent


def _make_tool_call_response(args_dict):
    """Helper to build a mock OpenAI tool-call response."""
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.arguments = json.dumps(args_dict)
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def _make_text_response(text):
    """Helper to build a mock OpenAI text response."""
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = text
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategySpec:
    """Tests for the StrategySpec Pydantic model."""

    def test_valid_spec(self):
        spec = StrategySpec(
            name="Test",
            description="A test strategy",
            strategy_type="trend_following",
            entry_rules=["buy when X"],
            exit_rules=["sell when Y"],
        )
        assert spec.name == "Test"
        assert spec.strategy_type == "trend_following"
        assert spec.risk_management == {}
        assert spec.params == {}

    def test_with_all_fields(self):
        spec = StrategySpec(
            name="Full",
            description="Full spec",
            strategy_type="momentum",
            entry_rules=["rule1", "rule2"],
            exit_rules=["exit1"],
            risk_management={"stop_loss": 0.05, "take_profit": 0.10},
            params={"period": 20},
        )
        assert spec.risk_management["stop_loss"] == 0.05
        assert spec.params["period"] == 20


class TestStrategyAgentInit:
    """Tests for StrategyAgent initialization."""

    @patch("strategy_agent.OpenAI")
    def test_default_model(self, mock_cls):
        agent = StrategyAgent(api_key="key")
        assert agent.model == "gpt-4o"
        mock_cls.assert_called_once_with(api_key="key")

    @patch("strategy_agent.OpenAI")
    def test_custom_model(self, mock_cls):
        agent = StrategyAgent(model="gpt-3.5-turbo", api_key="key")
        assert agent.model == "gpt-3.5-turbo"


class TestParseStrategy:
    """Tests for StrategyAgent.parse_strategy()."""

    def test_parse_strategy_success(self, mock_openai_client):
        args = {
            "name": "SMA Crossover",
            "description": "Buy when fast SMA crosses above slow SMA",
            "strategy_type": "trend_following",
            "entry_rules": ["fast SMA crosses above slow SMA"],
            "exit_rules": ["fast SMA crosses below slow SMA"],
            "risk_management": {"stop_loss": 0.05},
            "params": {"fast_period": 10, "slow_period": 30},
        }
        mock_openai_client.client.chat.completions.create.return_value = _make_tool_call_response(args)

        spec = mock_openai_client.parse_strategy("Buy when 10-day SMA crosses above 30-day SMA")

        assert isinstance(spec, StrategySpec)
        assert spec.name == "SMA Crossover"
        assert spec.strategy_type == "trend_following"
        assert len(spec.entry_rules) == 1
        assert spec.risk_management["stop_loss"] == 0.05

    def test_parse_strategy_no_tool_call(self, mock_openai_client):
        mock_openai_client.client.chat.completions.create.return_value = _make_text_response("some text")

        with pytest.raises(ValueError, match="did not return a function call"):
            mock_openai_client.parse_strategy("some strategy")

    def test_parse_strategy_calls_openai_with_correct_args(self, mock_openai_client):
        args = {
            "name": "Test",
            "description": "desc",
            "strategy_type": "mean_reversion",
            "entry_rules": ["r1"],
            "exit_rules": ["r2"],
        }
        mock_openai_client.client.chat.completions.create.return_value = _make_tool_call_response(args)

        mock_openai_client.parse_strategy("my strategy description")

        call_kwargs = mock_openai_client.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
        # Should have tool_choice specifying define_strategy
        assert call_kwargs.kwargs["tool_choice"]["function"]["name"] == "define_strategy"


class TestGenerateCode:
    """Tests for StrategyAgent.generate_code()."""

    def test_generate_code_returns_string(self, mock_openai_client):
        spec = StrategySpec(
            name="Test",
            description="desc",
            strategy_type="trend_following",
            entry_rules=["r1"],
            exit_rules=["r2"],
        )
        strategy_code = """
import backtrader as bt

class TestStrategy(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        pass
"""
        mock_openai_client.client.chat.completions.create.return_value = _make_text_response(strategy_code)

        code = mock_openai_client.generate_code(spec)
        assert "import backtrader" in code
        assert "class TestStrategy" in code

    def test_generate_code_strips_markdown_fences(self, mock_openai_client):
        spec = StrategySpec(
            name="Test",
            description="desc",
            strategy_type="trend_following",
            entry_rules=["r1"],
            exit_rules=["r2"],
        )
        fenced_code = "```python\nimport backtrader as bt\nclass Foo(bt.Strategy):\n    pass\n```"
        mock_openai_client.client.chat.completions.create.return_value = _make_text_response(fenced_code)

        code = mock_openai_client.generate_code(spec)
        assert not code.startswith("```")
        assert not code.endswith("```")
        assert "import backtrader" in code

    def test_generate_code_strips_plain_fences(self, mock_openai_client):
        spec = StrategySpec(
            name="Test",
            description="desc",
            strategy_type="trend_following",
            entry_rules=["r1"],
            exit_rules=["r2"],
        )
        fenced_code = "```\nimport backtrader as bt\n```"
        mock_openai_client.client.chat.completions.create.return_value = _make_text_response(fenced_code)

        code = mock_openai_client.generate_code(spec)
        assert "```" not in code


class TestValidateCode:
    """Tests for StrategyAgent.validate_code() delegation."""

    def test_valid_code(self, mock_openai_client):
        code = """
import backtrader as bt

class MyStrat(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        pass
"""
        is_valid, msg = mock_openai_client.validate_code(code)
        assert is_valid is True

    def test_invalid_code(self, mock_openai_client):
        code = "x = 1"
        is_valid, msg = mock_openai_client.validate_code(code)
        assert is_valid is False


class TestExecuteStrategy:
    """Tests for StrategyAgent.execute_strategy() with real backtest."""

    def test_execute_with_valid_strategy(self, mock_openai_client, sample_ohlcv_df, simple_ma_strategy_code):
        result = mock_openai_client.execute_strategy(simple_ma_strategy_code, sample_ohlcv_df)
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")

    def test_execute_with_invalid_code(self, mock_openai_client, sample_ohlcv_df):
        result = mock_openai_client.execute_strategy("not a strategy", sample_ohlcv_df)
        # Should return a result with error in trade_log
        assert result.total_return == 0.0


class TestRunPipeline:
    """Tests for StrategyAgent.run_pipeline() (integration with mocks)."""

    def test_pipeline_returns_all_keys(self, mock_openai_client, sample_ohlcv_df, simple_ma_strategy_code):
        """Full pipeline with mocked LLM returns expected dict keys."""
        # Mock parse_strategy
        parse_args = {
            "name": "SMA",
            "description": "desc",
            "strategy_type": "trend_following",
            "entry_rules": ["r1"],
            "exit_rules": ["r2"],
        }

        # First call is parse, second is generate
        mock_openai_client.client.chat.completions.create.side_effect = [
            _make_tool_call_response(parse_args),
            _make_text_response(simple_ma_strategy_code),
        ]

        result = mock_openai_client.run_pipeline("Buy SMA crossover", sample_ohlcv_df)

        assert "spec" in result
        assert "code" in result
        assert "is_valid" in result
        assert "validation_msg" in result
        assert "result" in result
        assert isinstance(result["spec"], StrategySpec)
        assert result["is_valid"] is True
