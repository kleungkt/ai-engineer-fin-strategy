"""Tests for the backtester module (backtester.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import (
    BacktestResult,
    BollingerBounce,
    build_strategy,
    MACDSignal,
    MACrossover,
    map_query_to_strategy,
    RSIExtreme,
    run_backtest,
)
from screener import generate_sample_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data() -> pd.DataFrame:
    """200 days of synthetic OHLCV data for backtesting."""
    return generate_sample_data("TEST", days=200)


@pytest.fixture()
def tiny_data() -> pd.DataFrame:
    """Only 10 rows – too little for many strategies."""
    return generate_sample_data("TINY", days=10)


@pytest.fixture()
def good_result() -> BacktestResult:
    """A hand-crafted BacktestResult representing strong performance."""
    return BacktestResult(
        total_return=45.0,
        annual_return=22.0,
        sharpe_ratio=1.8,
        max_drawdown=8.0,
        max_drawdown_duration=30,
        win_rate=62.0,
        total_trades=50,
        trade_log=[],
    )


# ---------------------------------------------------------------------------
# build_strategy – factory tests
# ---------------------------------------------------------------------------

class TestBuildStrategy:
    def test_ma_crossover_returns_subclass(self):
        cls = build_strategy("ma_crossover")
        assert issubclass(cls, MACrossover)

    def test_rsi_extreme_returns_subclass(self):
        cls = build_strategy("rsi_extreme")
        assert issubclass(cls, RSIExtreme)

    def test_macd_signal_returns_subclass(self):
        cls = build_strategy("macd_signal")
        assert issubclass(cls, MACDSignal)

    def test_bollinger_bounce_returns_subclass(self):
        cls = build_strategy("bollinger_bounce")
        assert issubclass(cls, BollingerBounce)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_strategy("nonexistent_strategy")

    def test_params_forwarded(self):
        """Building with custom params should not raise."""
        cls = build_strategy("ma_crossover", {"fast": 5, "slow": 20})
        assert issubclass(cls, MACrossover)

    def test_configured_name(self):
        cls = build_strategy("ma_crossover")
        assert "Configured" in cls.__name__


# ---------------------------------------------------------------------------
# run_backtest – integration tests
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_backtest_result(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover")
        assert isinstance(result, BacktestResult)

    def test_result_fields_populated(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover")
        assert isinstance(result.total_return, float)
        assert isinstance(result.annual_return, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.trade_log, list)

    def test_sharpe_ratio_is_float_or_none(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover")
        assert result.sharpe_ratio is None or isinstance(result.sharpe_ratio, float)

    def test_rsi_strategy_runs(self, sample_data):
        result = run_backtest(sample_data, "rsi_extreme")
        assert isinstance(result, BacktestResult)

    def test_macd_strategy_runs(self, sample_data):
        result = run_backtest(sample_data, "macd_signal")
        assert isinstance(result, BacktestResult)

    def test_bollinger_strategy_runs(self, sample_data):
        result = run_backtest(sample_data, "bollinger_bounce")
        assert isinstance(result, BacktestResult)

    def test_custom_initial_cash(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover", initial_cash=50_000)
        assert isinstance(result, BacktestResult)

    def test_custom_params(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover", params={"fast": 5, "slow": 15})
        assert isinstance(result, BacktestResult)

    def test_trade_log_entries_have_expected_keys(self, sample_data):
        result = run_backtest(sample_data, "ma_crossover")
        for entry in result.trade_log:
            assert "entry_date" in entry
            assert "entry_price" in entry
            # exit fields may or may not be present depending on open position
            if "exit_date" in entry:
                assert "exit_price" in entry
                assert "pnl" in entry

    def test_insufficient_data_handled_gracefully(self, tiny_data):
        """Should not crash even with very little data."""
        # 10 rows might produce 0 trades, but should still return a result
        result = run_backtest(tiny_data, "ma_crossover")
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0


# ---------------------------------------------------------------------------
# BacktestResult model
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_defaults(self):
        r = BacktestResult()
        assert r.total_return == 0.0
        assert r.annual_return == 0.0
        assert r.sharpe_ratio is None
        assert r.max_drawdown == 0.0
        assert r.win_rate == 0.0
        assert r.total_trades == 0
        assert r.trade_log == []

    def test_model_dump(self, good_result):
        d = good_result.model_dump()
        assert "total_return" in d
        assert "annual_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "win_rate" in d
        assert "total_trades" in d
        assert "trade_log" in d

    def test_values_match(self, good_result):
        assert good_result.total_return == 45.0
        assert good_result.sharpe_ratio == 1.8


# ---------------------------------------------------------------------------
# map_query_to_strategy
# ---------------------------------------------------------------------------

class TestMapQueryToStrategy:
    class _FakeQuery:
        def __init__(self, indicators=None, intent_type=""):
            self.indicators = indicators
            self.intent_type = intent_type

    def test_macd_indicator(self):
        q = self._FakeQuery(indicators=["MACD"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "macd_signal"

    def test_rsi_indicator(self):
        q = self._FakeQuery(indicators=["RSI_14"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "rsi_extreme"

    def test_bollinger_indicator(self):
        q = self._FakeQuery(indicators=["Bollinger"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "bollinger_bounce"

    def test_ma_indicator(self):
        q = self._FakeQuery(indicators=["MA_20"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "ma_crossover"

    def test_sma_indicator(self):
        q = self._FakeQuery(indicators=["SMA_50"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "ma_crossover"

    def test_ema_indicator(self):
        q = self._FakeQuery(indicators=["EMA_12"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "ma_crossover"

    def test_no_indicators_defaults_to_ma(self):
        q = self._FakeQuery(indicators=None)
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "ma_crossover"

    def test_unknown_indicator_defaults_to_ma(self):
        q = self._FakeQuery(indicators=["UNKNOWN_XYZ"])
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "ma_crossover"

    def test_macd_takes_priority_over_ma(self):
        """MACD should be detected even if MA is also present."""
        q = self._FakeQuery(indicators=["MA_20", "MACD"])
        strategy_type, _ = map_query_to_strategy(q)
        assert strategy_type == "macd_signal"

    def test_rsi_with_thresholds(self):
        q = self._FakeQuery(indicators=["RSI"], intent_type="buy when rsi below 25 above 75")
        strategy_type, params = map_query_to_strategy(q)
        assert strategy_type == "rsi_extreme"
        # 25 < 50 → oversold, 75 >= 50 → overbought
        assert params.get("oversold") == 25
        assert params.get("overbought") == 75

    def test_case_insensitive(self):
        q = self._FakeQuery(indicators=["macd"])
        strategy_type, _ = map_query_to_strategy(q)
        assert strategy_type == "macd_signal"
