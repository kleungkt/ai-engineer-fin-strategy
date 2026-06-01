"""Shared fixtures for AI Strategy Generator tests."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture
def sample_ohlcv_df():
    """Generate a small sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=200)
    initial_price = 100.0
    returns = np.random.normal(0.0002, 0.02, 200)
    prices = initial_price * np.cumprod(1 + returns)

    data = []
    for date, close in zip(dates, prices):
        daily_range = close * np.random.uniform(0.01, 0.04)
        open_price = close + np.random.uniform(-daily_range / 2, daily_range / 2)
        high = max(open_price, close) + np.random.uniform(0, daily_range / 2)
        low = min(open_price, close) - np.random.uniform(0, daily_range / 2)
        volume = np.random.uniform(1e6, 5e6)
        data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume, 0),
        })

    return pd.DataFrame(data)


@pytest.fixture
def simple_ma_strategy_code():
    """A simple Backtrader strategy code for testing."""
    return '''
import backtrader as bt


class SimpleMA(bt.Strategy):
    """Simple moving average crossover strategy."""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy(size=10)
        else:
            if self.crossover < 0:
                self.close()
'''


@pytest.fixture
def sample_strategy_spec():
    """Return a dict matching StrategySpec structure."""
    return {
        "name": "SMA Crossover",
        "description": "Buy when fast SMA crosses above slow SMA.",
        "strategy_type": "trend_following",
        "entry_rules": ["fast SMA crosses above slow SMA"],
        "exit_rules": ["fast SMA crosses below slow SMA"],
        "risk_management": {"stop_loss": 0.05},
        "params": {"fast_period": 10, "slow_period": 30},
    }
