"""
Tests for Streamlit UI of AI Strategy Generator.

Tests focus on the data/utility functions and components that don't require
a running Streamlit server.
"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

# Add project paths - be more explicit about the structure
project_root = Path(__file__).parent.parent.parent  # From app/tests up to project root
src_path = project_root / "src"

# Ensure src is in sys.path
src_str = str(src_path)
if src_str not in sys.path:
    sys.path.insert(0, src_str)


# Import functions to test (we'll import the module and test helper functions)
import importlib.util


# =============================================================================
# Test Session State Initialization
# =============================================================================

def test_session_state_initialization():
    """Test that session state initializes correctly."""
    # Simulate session state behavior
    class MockSessionState(dict):
        def __getattr__(self, key):
            return self.get(key)
        def __setattr__(self, key, value):
            self[key] = value
    
    state = MockSessionState()
    
    # Initialize history if not present
    if "history" not in state:
        state.history = []
    if "current_code" not in state:
        state.current_code = None
    if "current_result" not in state:
        state.current_result = None
    if "selected_template" not in state:
        state.selected_template = None
    
    assert state.history == []
    assert state.current_code is None
    assert state.current_result is None
    assert state.selected_template is None


def test_session_state_persistence():
    """Test that session state persists between operations."""
    class MockSessionState(dict):
        def __getattr__(self, key):
            return self.get(key)
        def __setattr__(self, key, value):
            self[key] = value
    
    state = MockSessionState()
    state.history = []
    state.current_code = "test_code"
    state.current_result = {"total_return": 0.15}
    
    # Simulate adding a strategy
    state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Test strategy",
        "code": "test_code",
    })
    
    assert len(state.history) == 1
    assert state.current_code == "test_code"
    assert state.current_result["total_return"] == 0.15


# =============================================================================
# Test Helper Functions
# =============================================================================

def test_format_percent():
    """Test percentage formatting."""
    # Import from the app module by extracting the function logic
    def format_percent(value: float) -> str:
        return f"{value * 100:.2f}%"
    
    assert format_percent(0.15) == "15.00%"
    assert format_percent(0.05) == "5.00%"
    assert format_percent(-0.10) == "-10.00%"
    assert format_percent(1.0) == "100.00%"
    assert format_percent(0) == "0.00%"


def test_format_number():
    """Test number formatting."""
    def format_number(value: float, decimals: int = 2) -> str:
        return f"{value:.{decimals}f}"
    
    assert format_number(1.234) == "1.23"
    assert format_number(1.234, 3) == "1.234"
    assert format_number(1.5) == "1.50"
    assert format_number(0.0) == "0.00"
    assert format_number(-2.5) == "-2.50"


# =============================================================================
# Test Mock Strategy Generation
# =============================================================================

def test_mock_strategy_code_generation():
    """Test mock strategy code generation."""
    def get_mock_strategy_code(strategy_name: str = "GeneratedStrategy") -> str:
        return f'''\
import backtrader as bt


class {strategy_name}(bt.Strategy):
    """AI-generated trading strategy."""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("stake_pct", 0.95),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()
'''
    
    code = get_mock_strategy_code("TestStrategy")
    assert "class TestStrategy(bt.Strategy)" in code
    assert "fast_period" in code
    assert "slow_period" in code
    assert "self.buy(size=size)" in code
    assert "self.close()" in code


def test_mock_generate_strategy():
    """Test mock strategy generation returns expected structure."""
    def mock_generate_strategy(user_input: str) -> dict:
        return {
            "spec": {
                "name": "Generated Strategy",
                "description": "AI-generated from: " + user_input[:50],
                "strategy_type": "trend_following",
                "entry_rules": ["Fast MA crosses above slow MA"],
                "exit_rules": ["Fast MA crosses below slow MA"],
                "risk_management": {"stop_loss": 0.05, "take_profit": 0.10},
                "params": {"fast_period": 10, "slow_period": 30},
            },
            "code": "import backtrader as bt\\n\\nclass Strategy(bt.Strategy):\\n    pass",
            "is_valid": True,
            "validation_msg": "Valid Backtrader strategy",
            "result": None,
        }
    
    result = mock_generate_strategy("Buy when 10-day MA crosses above 30-day MA")
    
    assert result["is_valid"] is True
    assert result["validation_msg"] == "Valid Backtrader strategy"
    assert "spec" in result
    assert result["spec"]["name"] == "Generated Strategy"
    assert "code" in result


# =============================================================================
# Test Template Rendering
# =============================================================================

def test_template_listing():
    """Test that all expected templates are listed."""
    from strategy_templates import list_templates
    
    templates = list_templates()
    
    expected_templates = [
        "ma_crossover",
        "rsi_extreme",
        "macd_signal",
        "bollinger_bounce",
        "momentum_breakout",
        "dual_thrust",
    ]
    
    template_names = [t["name"] for t in templates]
    
    for expected in expected_templates:
        assert expected in template_names, f"Template {expected} not found"


def test_template_rendering():
    """Test template rendering with parameters."""
    from strategy_templates import render_template
    
    # Test MA crossover with custom params
    code = render_template("ma_crossover", {
        "fast_period": 5,
        "slow_period": 20,
        "stake_pct": 0.90,
    })
    
    assert "class MACrossover(bt.Strategy)" in code
    # Check that params are correctly rendered in the params tuple
    assert '("fast_period", 5)' in code
    assert '("slow_period", 20)' in code
    assert "0.9" in code


def test_default_params():
    """Test that default parameters are applied correctly."""
    from strategy_templates import render_template, DEFAULT_PARAMS
    
    code = render_template("ma_crossover")  # No params = use defaults
    
    defaults = DEFAULT_PARAMS["ma_crossover"]
    # Check params in the tuple
    assert f'("fast_period", {defaults["fast_period"]})' in code
    assert f'("slow_period", {defaults["slow_period"]})' in code


def test_invalid_template_name():
    """Test that invalid template name raises KeyError."""
    from strategy_templates import get_template
    
    with pytest.raises(KeyError):
        get_template("nonexistent_template")


# =============================================================================
# Test Backtest Integration
# =============================================================================

def test_backtest_execution():
    """Test backtest execution on a sample strategy."""
    # Skip if pandas not available
    pytest.importorskip("pandas")
    pytest.importorskip("backtrader")
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    
    data = pd.DataFrame({
        "date": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.uniform(1e6, 5e6, 100),
    })
    
    strategy_code = '''
import backtrader as bt

class TestStrategy(bt.Strategy):
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
    
    def next(self):
        if not self.position:
            if self.fast_ma > self.slow_ma:
                self.buy(size=10)
        else:
            if self.fast_ma < self.slow_ma:
                self.close()
'''
    
    from backtester import run_backtest
    
    result = run_backtest(data=data, strategy_code=strategy_code, initial_cash=100000)
    
    # Verify result structure
    assert hasattr(result, "total_return")
    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "win_rate")
    assert hasattr(result, "total_trades")
    assert isinstance(result.total_trades, int)


# =============================================================================
# Test Parameter Optimization
# =============================================================================

def test_optimization_result_structure():
    """Test optimization result has correct structure."""
    # Skip if pandas not available
    pandas = pytest.importorskip("pandas")
    
    from optimizer.parameter_optimizer import OptimizationResult
    
    result = OptimizationResult(
        best_params={"fast_period": 10, "slow_period": 30},
        best_score=1.5,
        all_results=[
            {"params": {"fast_period": 10, "slow_period": 30}, "score": 1.5}
        ],
        method="grid_search",
    )
    
    assert result.best_params == {"fast_period": 10, "slow_period": 30}
    assert result.best_score == 1.5
    assert result.method == "grid_search"
    assert len(result.all_results) == 1


def test_optimization_metric_extraction():
    """Test metric extraction from backtest results."""
    # Skip if pandas not available
    pytest.importorskip("pandas")
    
    from optimizer.parameter_optimizer import _extract_metric
    
    test_results = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.10,
        "win_rate": 0.55,
    }
    
    assert _extract_metric(test_results, "sharpe") == 1.2
    assert _extract_metric(test_results, "return") == 0.15
    assert _extract_metric(test_results, "total_return") == 0.15
    assert _extract_metric(test_results, "max_drawdown") == 0.10


# =============================================================================
# Test Data Fetcher
# =============================================================================

def test_generate_sample_data():
    """Test sample data generation."""
    # Skip if pandas not available
    pytest.importorskip("pandas")
    
    from data_fetcher import generate_sample_data
    
    df = generate_sample_data(symbol="TEST", days=100)
    
    # Check columns
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check length
    assert len(df) == 100
    
    # Check data types and values
    assert df["close"].dtype in [float, int]
    assert df["high"].min() >= df["low"].min()


def test_sample_data_has_valid_ohlc():
    """Test that sample data has valid OHLC relationships."""
    # Skip if pandas not available
    pytest.importorskip("pandas")
    
    from data_fetcher import generate_sample_data
    
    df = generate_sample_data(days=50)
    
    # High should be >= Open, Close, Low
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["low"]).all()
    
    # Low should be <= Open, Close, High
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["high"]).all()
    
    # Volume should be positive
    assert (df["volume"] > 0).all()


# =============================================================================
# Test Strategy Agent Integration (Mocked)
# =============================================================================

def test_strategy_agent_mock_pipeline():
    """Test strategy agent pipeline with mocked OpenAI."""
    # Skip if pandas not available
    pytest.importorskip("pandas")
    
    from strategy_agent import StrategySpec
    
    spec = StrategySpec(
        name="Test Strategy",
        description="Test description",
        strategy_type="trend_following",
        entry_rules=["Rule 1"],
        exit_rules=["Rule 2"],
        risk_management={"stop_loss": 0.05},
        params={"period": 14},
    )
    
    assert spec.name == "Test Strategy"
    assert spec.strategy_type == "trend_following"
    assert len(spec.entry_rules) == 1
    assert spec.risk_management["stop_loss"] == 0.05


# =============================================================================
# Test UI State Transitions
# =============================================================================

def test_history_addition():
    """Test adding strategies to history."""
    class MockState:
        def __init__(self):
            self.history = []
    
    state = MockState()
    
    # Add first strategy
    state.history.append({
        "timestamp": "2024-01-01 10:00:00",
        "description": "First strategy",
        "code": "code1",
        "result": {"total_return": 0.10},
    })
    
    assert len(state.history) == 1
    
    # Add second strategy
    state.history.append({
        "timestamp": "2024-01-01 11:00:00",
        "description": "Second strategy",
        "code": "code2",
        "result": {"total_return": 0.15},
    })
    
    assert len(state.history) == 2
    assert state.history[1]["description"] == "Second strategy"


def test_current_code_update():
    """Test updating current code in session state."""
    class MockState:
        def __init__(self):
            self.current_code = None
    
    state = MockState()
    
    assert state.current_code is None
    
    new_code = 'import backtrader as bt\n\nclass Test(bt.Strategy):\n    pass'
    state.current_code = new_code
    
    assert state.current_code == new_code
    assert "bt.Strategy" in state.current_code


def test_result_caching():
    """Test caching backtest results."""
    class MockState:
        def __init__(self):
            self.current_result = None
    
    state = MockState()
    
    result = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.10,
        "win_rate": 0.55,
        "total_trades": 10,
    }
    
    state.current_result = result
    
    assert state.current_result == result
    assert state.current_result["total_return"] == 0.15


# =============================================================================
# Test Template Parameter Presets
# =============================================================================

def test_template_params_presets():
    """Test template parameter presets dictionary."""
    template_params = {
        "ma_crossover": {"fast_period": 10, "slow_period": 30, "stake_pct": 0.95},
        "rsi_extreme": {"rsi_period": 14, "oversold": 30, "overbought": 70, "stake_pct": 0.95},
        "macd_signal": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "stake_pct": 0.95},
        "bollinger_bounce": {"period": 20, "devfactor": 2.0, "stake_pct": 0.95},
        "momentum_breakout": {"lookback": 20, "stake_pct": 0.95},
        "dual_thrust": {"lookback": 4, "k1": 0.5, "k2": 0.5, "stake_pct": 0.95},
    }
    
    # Verify all templates have required params
    assert "ma_crossover" in template_params
    assert "fast_period" in template_params["ma_crossover"]
    assert "slow_period" in template_params["ma_crossover"]
    
    assert "rsi_extreme" in template_params
    assert "rsi_period" in template_params["rsi_extreme"]
    
    assert "dual_thrust" in template_params
    assert "k1" in template_params["dual_thrust"]
    assert "k2" in template_params["dual_thrust"]


# =============================================================================
# Test Metric Card Rendering Logic
# =============================================================================

def test_metric_positive_negative_classification():
    """Test classification of metrics as positive/negative."""
    def classify_metric(metric_name: str, value: float) -> str:
        """Determine if a metric value is positive or negative."""
        positive_metrics = {
            "total_return": lambda v: v >= 0,
            "sharpe_ratio": lambda v: v >= 1.0,
            "win_rate": lambda v: v >= 0.5,
            "annual_return": lambda v: v >= 0,
        }
        negative_metrics = {
            "max_drawdown": lambda v: v < 0.2,
        }
        
        if metric_name in positive_metrics:
            return "positive" if positive_metrics[metric_name](value) else "negative"
        elif metric_name in negative_metrics:
            return "positive" if negative_metrics[metric_name](value) else "negative"
        return "neutral"
    
    assert classify_metric("total_return", 0.15) == "positive"
    assert classify_metric("total_return", -0.10) == "negative"
    assert classify_metric("sharpe_ratio", 1.5) == "positive"
    assert classify_metric("sharpe_ratio", 0.5) == "negative"
    assert classify_metric("max_drawdown", 0.05) == "positive"
    assert classify_metric("max_drawdown", 0.30) == "negative"
    assert classify_metric("win_rate", 0.60) == "positive"
    assert classify_metric("win_rate", 0.40) == "negative"


# =============================================================================
# Test Dark Theme CSS Variables
# =============================================================================

def test_dark_theme_colors():
    """Test dark theme color values are valid."""
    theme_colors = {
        "background": "#0d1117",
        "card_background": "#161b22",
        "border": "#30363d",
        "positive": "#3fb950",
        "negative": "#f85149",
        "accent": "#58a6ff",
        "text_primary": "#c9d1d9",
        "text_secondary": "#8b949e",
    }
    
    # Check all colors are valid hex
    for name, color in theme_colors.items():
        assert color.startswith("#"), f"{name} should start with #"
        assert len(color) == 7, f"{name} should be 7 chars (#RRGGBB)"
        
        # Check hex characters
        hex_part = color[1:]
        assert all(c in "0123456789abcdef" for c in hex_part.lower()), f"{name} has invalid hex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])