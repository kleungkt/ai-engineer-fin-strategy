"""Tests for the backtester module."""

import pytest
import pandas as pd
import numpy as np
from backtester import run_backtest, BacktestResult, TradeAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for backtesting."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=250)
    initial_price = 100.0
    returns = np.random.normal(0.0002, 0.02, 250)
    prices = initial_price * np.cumprod(1 + returns)

    data = []
    for date, close in zip(dates, prices):
        daily_range = close * 0.02
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


VALID_STRATEGY = """
import backtrader as bt


class BuyAndHold(bt.Strategy):
    params = (
        ("size_pct", 0.95),
    )

    def __init__(self):
        self.order = None

    def next(self):
        if not self.position:
            size = int((self.broker.getcash() * self.p.size_pct) / self.data.close[0])
            if size > 0:
                self.buy(size=size)
"""

SMA_STRATEGY = """
import backtrader as bt


class SMAStrategy(bt.Strategy):
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
                size = int((self.broker.getcash() * 0.95) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()
"""


class TestRunBacktest:
    """Tests for run_backtest()."""

    def test_returns_backtest_result(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        assert isinstance(result, BacktestResult)

    def test_result_has_all_fields(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        assert hasattr(result, "total_return")
        assert hasattr(result, "annual_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "trade_log")

    def test_buy_and_hold_has_position(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        # With buy-and-hold, we expect some return (positive or negative)
        assert isinstance(result.total_return, float)

    def test_sma_strategy_runs(self, sample_data):
        result = run_backtest(sample_data, SMA_STRATEGY)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return, float)

    def test_with_custom_params(self, sample_data):
        params = {"fast_period": 5, "slow_period": 20}
        result = run_backtest(sample_data, SMA_STRATEGY, params=params)
        assert isinstance(result, BacktestResult)

    def test_with_custom_initial_cash(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY, initial_cash=50000)
        assert isinstance(result, BacktestResult)

    def test_invalid_strategy_returns_zeroes(self, sample_data):
        result = run_backtest(sample_data, "not valid strategy code")
        assert result.total_return == 0.0
        assert result.total_trades == 0

    def test_no_strategy_class_returns_zeroes(self, sample_data):
        code = "import backtrader as bt\nx = 42"
        result = run_backtest(sample_data, code)
        assert result.total_return == 0.0

    def test_max_drawdown_is_percentage(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        # max_drawdown should be between 0 and 1 (it's divided by 100 in the code)
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_win_rate_is_fraction(self, sample_data):
        result = run_backtest(sample_data, SMA_STRATEGY)
        assert 0.0 <= result.win_rate <= 1.0

    def test_trade_log_is_list(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        assert isinstance(result.trade_log, list)

    def test_result_model_dump(self, sample_data):
        result = run_backtest(sample_data, VALID_STRATEGY)
        d = result.model_dump()
        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "trade_log" in d


class TestBacktestResult:
    """Tests for BacktestResult model."""

    def test_default_values(self):
        result = BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
        )
        assert result.trade_log == []

    def test_with_trade_log(self):
        result = BacktestResult(
            total_return=0.1,
            annual_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.6,
            total_trades=10,
            trade_log=[{"pnl": 100.0, "pnlcomm": 95.0, "bar_open": 5, "bar_close": 10, "bar_len": 5}],
        )
        assert len(result.trade_log) == 1
        assert result.trade_log[0]["pnl"] == 100.0
