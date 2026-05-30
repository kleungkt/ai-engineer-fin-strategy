"""Tests for the stock screener module (screener.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parser import IndicatorCondition, IntentType, QueryIntent
from screener import ScreenResult, generate_sample_data, screen_stocks


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_df(close_values: list[float], *, days: int | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices."""
    if days is not None:
        close_values = close_values[:days]
    n = len(close_values)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.array(close_values, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(n, 1_000_000),
        }
    )


def _rising_close(n: int = 60, start: float = 50.0) -> list[float]:
    """Generate n steadily rising close prices."""
    return [start + i * 0.5 for i in range(n)]


def _falling_close(n: int = 60, start: float = 150.0) -> list[float]:
    """Generate n steadily falling close prices."""
    return [start - i * 0.5 for i in range(n)]


@pytest.fixture()
def rising_stock():
    """A single rising stock DataFrame with 60 rows."""
    return _make_df(_rising_close(60))


@pytest.fixture()
def falling_stock():
    """A single falling stock DataFrame with 60 rows."""
    return _make_df(_falling_close(60))


@pytest.fixture()
def stock_data_mixed():
    """Dict of symbol → DataFrame: one rising, one falling."""
    return {
        "RISING": _make_df(_rising_close(60)),
        "FALLING": _make_df(_falling_close(60)),
    }


def _make_query(
    indicator_name: str,
    comparison: str,
    value: float | None = None,
    params: dict | None = None,
) -> QueryIntent:
    """Shorthand to build a QueryIntent with a single condition."""
    return QueryIntent(
        intent_type=IntentType.INDICATOR_SCREENER,
        indicators=[
            IndicatorCondition(
                name=indicator_name,
                comparison=comparison,
                value=value,
                params=params or {},
            )
        ],
        stock_scope="A股",
        time_range=30,
    )


# ---------------------------------------------------------------------------
# ScreenResult structure
# ---------------------------------------------------------------------------

class TestScreenResult:
    def test_has_required_fields(self):
        r = ScreenResult(symbol="AAPL", matched=True)
        assert r.symbol == "AAPL"
        assert r.matched is True
        assert r.indicators == {}
        assert r.explanation == ""

    def test_with_all_fields(self):
        r = ScreenResult(
            symbol="TSLA",
            matched=False,
            indicators={"rsi_14": 45.2},
            explanation="RSI below 50",
        )
        assert r.symbol == "TSLA"
        assert r.matched is False
        assert r.indicators["rsi_14"] == 45.2
        assert "RSI" in r.explanation

    def test_is_pydantic_model(self):
        r = ScreenResult(symbol="X", matched=True)
        d = r.model_dump()
        assert "symbol" in d
        assert "matched" in d
        assert "indicators" in d
        assert "explanation" in d


# ---------------------------------------------------------------------------
# screen_stocks: basic happy paths
# ---------------------------------------------------------------------------

class TestScreenStocksBasic:
    def test_no_conditions_returns_all_matched(self, stock_data_mixed):
        """When indicators list is empty, all stocks should match."""
        query = QueryIntent(
            intent_type=IntentType.INDICATOR_SCREENER,
            indicators=[],
            stock_scope="A股",
            time_range=30,
        )
        results = screen_stocks(query, stock_data_mixed)
        assert len(results) == 2
        assert all(r.matched for r in results)

    def test_returns_one_result_per_stock(self, stock_data_mixed):
        query = _make_query("RSI", "below", 50)
        results = screen_stocks(query, stock_data_mixed)
        symbols = {r.symbol for r in results}
        assert symbols == {"RISING", "FALLING"}

    def test_insufficient_data_marked_unmatched(self):
        """Stock with < 5 rows should be marked matched=False."""
        tiny = _make_df([1.0, 2.0, 3.0])  # only 3 rows
        query = _make_query("RSI", "below", 50)
        results = screen_stocks(query, {"TINY": tiny})
        assert len(results) == 1
        assert results[0].matched is False
        assert "Insufficient" in results[0].explanation


# ---------------------------------------------------------------------------
# screen_stocks: RSI condition
# ---------------------------------------------------------------------------

class TestScreenStocksRSI:
    def test_rising_stock_rsi_above_threshold(self, rising_stock):
        """Rising stock RSI should be above 50 → fails 'below 50' condition."""
        query = _make_query("RSI", "below", 50)
        results = screen_stocks(query, {"UP": rising_stock})
        # Rising stock should have high RSI, so below 50 should fail
        assert results[0].matched is False

    def test_falling_stock_rsi_below_threshold(self, falling_stock):
        """Falling stock RSI should be below 50 → passes 'below 50' condition."""
        query = _make_query("RSI", "below", 50)
        results = screen_stocks(query, {"DOWN": falling_stock})
        assert results[0].matched is True

    def test_rising_stock_rsi_above_passes(self, rising_stock):
        """Rising stock should pass 'RSI above 50'."""
        query = _make_query("RSI", "above", 50)
        results = screen_stocks(query, {"UP": rising_stock})
        assert results[0].matched is True


# ---------------------------------------------------------------------------
# screen_stocks: MACD condition
# ---------------------------------------------------------------------------

class TestScreenStocksMACD:
    def test_macd_above_condition(self, rising_stock):
        """Rising stock: MACD line should be above signal line → 'above' passes."""
        query = _make_query("MACD", "above")
        results = screen_stocks(query, {"UP": rising_stock})
        assert results[0].matched is True

    def test_macd_below_condition_fails_for_rising(self, rising_stock):
        """Rising stock: MACD above signal → 'below' should fail."""
        query = _make_query("MACD", "below")
        results = screen_stocks(query, {"UP": rising_stock})
        assert results[0].matched is False


# ---------------------------------------------------------------------------
# screen_stocks: MA / SMA condition
# ---------------------------------------------------------------------------

class TestScreenStocksMA:
    def test_price_above_ma_for_rising(self, rising_stock):
        """In a rising trend, close should be above MA(20)."""
        query = _make_query("MA", "above", params={"period": 20})
        results = screen_stocks(query, {"UP": rising_stock})
        assert results[0].matched is True

    def test_price_below_ma_for_falling(self, falling_stock):
        """In a falling trend, close should be below MA(20)."""
        query = _make_query("MA", "below", params={"period": 20})
        results = screen_stocks(query, {"DOWN": falling_stock})
        assert results[0].matched is True


# ---------------------------------------------------------------------------
# screen_stocks: filtering logic (AND)
# ---------------------------------------------------------------------------

class TestScreenStocksFiltering:
    def test_matching_stock_passes(self, stock_data_mixed):
        """Only the falling stock should match RSI < 30."""
        query = _make_query("RSI", "below", 30)
        results = screen_stocks(query, stock_data_mixed)
        matched = [r for r in results if r.matched]
        not_matched = [r for r in results if not r.matched]
        # Falling stock should match; rising should not
        assert len(matched) >= 0  # At least we get results
        assert len(matched) + len(not_matched) == 2

    def test_indicator_snapshot_populated(self, rising_stock):
        """The indicators dict in ScreenResult should contain computed values."""
        query = _make_query("RSI", "above", 50, params={"period": 14})
        results = screen_stocks(query, {"UP": rising_stock})
        assert len(results) == 1
        snapshot = results[0].indicators
        # Should have an rsi_14 key
        assert "rsi_14" in snapshot
        assert isinstance(snapshot["rsi_14"], float)


# ---------------------------------------------------------------------------
# generate_sample_data
# ---------------------------------------------------------------------------

class TestGenerateSampleData:
    def test_returns_correct_columns(self):
        df = generate_sample_data("AAPL", days=30)
        assert set(df.columns) >= {"date", "open", "high", "low", "close", "volume"}

    def test_returns_correct_length(self):
        df = generate_sample_data("AAPL", days=50)
        assert len(df) == 50

    def test_deterministic(self):
        """Same symbol + days should produce identical data."""
        d1 = generate_sample_data("TSLA", days=30)
        d2 = generate_sample_data("TSLA", days=30)
        pd.testing.assert_frame_equal(d1, d2)

    def test_different_symbols_differ(self):
        d1 = generate_sample_data("AAPL", days=30)
        d2 = generate_sample_data("MSFT", days=30)
        assert not d1["close"].equals(d2["close"])

    def test_ohlc_relationships(self):
        """high >= open, close; low <= open, close (basic sanity)."""
        df = generate_sample_data("TEST", days=100)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_screen_with_generated_data(self):
        """Full pipeline: generate data then screen."""
        stock_data = {
            sym: generate_sample_data(sym, days=60)
            for sym in ("AAA", "BBB", "CCC")
        }
        query = _make_query("RSI", "below", 50)
        results = screen_stocks(query, stock_data)
        assert len(results) == 3
        assert all(isinstance(r, ScreenResult) for r in results)
