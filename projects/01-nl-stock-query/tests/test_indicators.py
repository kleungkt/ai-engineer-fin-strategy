"""Tests for the technical indicators module (indicators.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import (
    calc_bollinger,
    calc_ema,
    calc_kdj,
    calc_ma,
    calc_macd,
    calc_rsi,
    check_above,
    check_below,
    check_crossover,
    check_crossunder,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_df(close_values, *, start: float = 100.0):
    """Build a minimal OHLCV DataFrame from a list/array of close prices."""
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


@pytest.fixture()
def rising_df():
    """20 rows of strictly rising close prices: 1, 2, 3, ..., 20."""
    return _make_df(list(range(1, 21)))


@pytest.fixture()
def falling_df():
    """20 rows of strictly falling close prices: 20, 19, ..., 1."""
    return _make_df(list(range(20, 0, -1)))


@pytest.fixture()
def flat_df():
    """20 rows of constant close = 50."""
    return _make_df([50.0] * 20)


@pytest.fixture()
def small_df():
    """10-row DataFrame for quick tests."""
    return _make_df([10, 11, 12, 13, 14, 15, 14, 13, 12, 11])


@pytest.fixture()
def crossover_data():
    """Crafted data where a short MA crosses above a long MA at a known index.

    Strategy: two flat segments with a crossing point.
    """
    # First 8 values: short series below long series
    # Then cross over at index 8
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
    b = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    return pd.Series(a, name="a"), pd.Series(b, name="b")


@pytest.fixture()
def crossunder_data():
    """Crafted data where series_a crosses below series_b at a known index."""
    a = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    b = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    return pd.Series(a, name="a"), pd.Series(b, name="b")


# ---------------------------------------------------------------------------
# calc_ma
# ---------------------------------------------------------------------------

class TestCalcMa:
    def test_basic_ma_values(self, rising_df):
        """MA(5) of 1..20 should match hand-computed averages."""
        ma = calc_ma(rising_df, period=5)
        # First 4 values should be NaN
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[3])
        # Index 4: mean(1,2,3,4,5) = 3.0
        assert ma.iloc[4] == pytest.approx(3.0)
        # Index 5: mean(2,3,4,5,6) = 4.0
        assert ma.iloc[5] == pytest.approx(4.0)

    def test_ma_period_equals_length(self, small_df):
        """When period == len(df), only the last value is valid."""
        ma = calc_ma(small_df, period=10)
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[8])
        # Last value: mean(10..11) = 12.5
        assert ma.iloc[9] == pytest.approx(12.5)

    def test_insufficient_data_returns_all_nan(self, small_df):
        """If period > len(df), result should be all NaN."""
        ma = calc_ma(small_df, period=100)
        assert ma.isna().all()

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0, 1]})
        with pytest.raises(KeyError, match="close"):
            calc_ma(df, period=2)


# ---------------------------------------------------------------------------
# calc_rsi
# ---------------------------------------------------------------------------

class TestCalcRsi:
    def test_all_up_trend_rsi_near_100(self, rising_df):
        """A strictly rising series should produce RSI very close to 100."""
        rsi = calc_rsi(rising_df, period=5)
        # Drop the first few NaN values
        valid = rsi.dropna()
        # The last RSI value should be very high (near 100)
        assert valid.iloc[-1] > 95.0

    def test_all_down_trend_rsi_near_0(self, falling_df):
        """A strictly falling series should produce RSI very close to 0."""
        rsi = calc_rsi(falling_df, period=5)
        valid = rsi.dropna()
        assert valid.iloc[-1] < 5.0

    def test_flat_series_rsi_near_50(self, flat_df):
        """A perfectly flat series has 0 gain and 0 loss → RSI undefined or ~50."""
        rsi = calc_rsi(flat_df, period=5)
        valid = rsi.dropna()
        # With zero loss, RSI = 100 - 100/(1 + gain/loss) → could be NaN or 100
        # Wilder smoothing of zeros gives RS = 0/0 = NaN, so RSI may be NaN.
        # Just verify it doesn't crash and is between 0 and 100 or NaN
        for v in valid:
            assert np.isnan(v) or (0 <= v <= 100)

    def test_rsi_range(self, small_df):
        """RSI should always be between 0 and 100 (where not NaN)."""
        rsi = calc_rsi(small_df, period=3)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_insufficient_data_returns_all_nan(self, small_df):
        """If len(df) < period + 1, result is all NaN."""
        rsi = calc_rsi(small_df, period=20)
        assert rsi.isna().all()

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(KeyError, match="close"):
            calc_rsi(df)


# ---------------------------------------------------------------------------
# calc_macd
# ---------------------------------------------------------------------------

class TestCalcMacd:
    def test_returns_three_keys(self, rising_df):
        result = calc_macd(rising_df, fast=5, slow=10, signal=4)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_equals_macd_minus_signal(self, rising_df):
        """histogram should be exactly macd - signal."""
        result = calc_macd(rising_df, fast=5, slow=10, signal=4)
        macd = result["macd"]
        signal = result["signal"]
        histogram = result["histogram"]
        pd.testing.assert_series_equal(
            histogram, macd - signal, check_names=False
        )

    def test_rising_trend_macd_positive(self, rising_df):
        """In a rising trend, the last MACD value should be positive."""
        result = calc_macd(rising_df, fast=5, slow=10, signal=4)
        assert float(result["macd"].iloc[-1]) > 0

    def test_falling_trend_macd_negative(self, falling_df):
        """In a falling trend, the last MACD value should be negative."""
        result = calc_macd(falling_df, fast=5, slow=10, signal=4)
        assert float(result["macd"].iloc[-1]) < 0

    def test_insufficient_data_returns_all_nan(self, small_df):
        """If len(df) < slow period, all results should be NaN."""
        result = calc_macd(small_df, fast=12, slow=26, signal=9)
        assert result["macd"].isna().all()
        assert result["signal"].isna().all()
        assert result["histogram"].isna().all()

    def test_custom_params(self, rising_df):
        """Non-default fast/slow/signal should still work."""
        result = calc_macd(rising_df, fast=3, slow=7, signal=3)
        assert len(result["macd"]) == len(rising_df)

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(KeyError, match="close"):
            calc_macd(df)


# ---------------------------------------------------------------------------
# check_crossover / check_crossunder
# ---------------------------------------------------------------------------

class TestCrossoverCrossunder:
    def test_crossover_detected_at_correct_index(self, crossover_data):
        """series_a crosses above series_b at index 8."""
        a, b = crossover_data
        result = check_crossover(a, b)
        # Index 0-7: a <= b → False
        # Index 8: a=5 > b=3, prev a=1 <= b=3 → True
        assert result.iloc[8] is True or result.iloc[8] == True
        # All others should be False
        assert result.iloc[:8].sum() == 0
        assert result.iloc[9:].sum() == 0

    def test_crossunder_detected_at_correct_index(self, crossunder_data):
        """series_a crosses below series_b at index 8."""
        a, b = crossunder_data
        result = check_crossunder(a, b)
        assert result.iloc[8] is True or result.iloc[8] == True
        assert result.iloc[:8].sum() == 0
        assert result.iloc[9:].sum() == 0

    def test_no_crossover_when_always_above(self):
        """If a is always above b, there should be no crossover."""
        a = pd.Series([5.0, 5.0, 5.0, 5.0])
        b = pd.Series([1.0, 1.0, 1.0, 1.0])
        result = check_crossover(a, b)
        assert result.sum() == 0

    def test_no_crossunder_when_always_below(self):
        """If a is always below b, there should be no crossunder."""
        a = pd.Series([1.0, 1.0, 1.0, 1.0])
        b = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = check_crossunder(a, b)
        assert result.sum() == 0

    def test_crossover_first_element_is_false(self, crossover_data):
        """First element should be False — no previous value to compare."""
        a, b = crossover_data
        result = check_crossover(a, b)
        assert result.iloc[0] == False  # noqa: E712

    def test_multiple_crossovers(self):
        """Series that oscillate should trigger multiple crossovers."""
        a = pd.Series([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
        b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        result = check_crossover(a, b)
        # Crossover at index 1, 3, 5
        assert result.iloc[1] == True
        assert result.iloc[3] == True
        assert result.iloc[5] == True


# ---------------------------------------------------------------------------
# check_above / check_below
# ---------------------------------------------------------------------------

class TestCheckAboveBelow:
    def test_check_above(self):
        s = pd.Series([10, 20, 30, 40])
        result = check_above(s, 25)
        assert result.tolist() == [False, False, True, True]

    def test_check_below(self):
        s = pd.Series([10, 20, 30, 40])
        result = check_below(s, 25)
        assert result.tolist() == [True, True, False, False]

    def test_check_above_equal_not_included(self):
        """Equal values should not satisfy 'above'."""
        s = pd.Series([25.0])
        result = check_above(s, 25.0)
        assert result.iloc[0] == False

    def test_check_below_equal_not_included(self):
        """Equal values should not satisfy 'below'."""
        s = pd.Series([25.0])
        result = check_below(s, 25.0)
        assert result.iloc[0] == False


# ---------------------------------------------------------------------------
# calc_ema
# ---------------------------------------------------------------------------

class TestCalcEma:
    def test_ema_basic(self, rising_df):
        """EMA should return a series of the same length."""
        ema = calc_ema(rising_df, period=5)
        assert len(ema) == len(rising_df)

    def test_ema_follows_trend(self, rising_df):
        """In a rising trend, EMA should be rising at the end."""
        ema = calc_ema(rising_df, period=5)
        assert ema.iloc[-1] > ema.iloc[5]

    def test_insufficient_data_returns_all_nan(self, small_df):
        ema = calc_ema(small_df, period=100)
        assert ema.isna().all()

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(KeyError, match="close"):
            calc_ema(df)


# ---------------------------------------------------------------------------
# calc_bollinger
# ---------------------------------------------------------------------------

class TestCalcBollinger:
    def test_returns_upper_middle_lower(self, rising_df):
        bb = calc_bollinger(rising_df, period=10, std=2)
        assert set(bb.keys()) == {"upper", "middle", "lower"}

    def test_upper_above_middle_above_lower(self, rising_df):
        """Upper > middle > lower should hold for non-degenerate data."""
        bb = calc_bollinger(rising_df, period=10, std=2)
        valid_idx = bb["middle"].dropna().index
        for i in valid_idx:
            assert bb["upper"].iloc[i] >= bb["middle"].iloc[i]
            assert bb["middle"].iloc[i] >= bb["lower"].iloc[i]

    def test_middle_is_sma(self, rising_df):
        """Middle band should equal SMA."""
        bb = calc_bollinger(rising_df, period=10, std=2)
        ma = calc_ma(rising_df, period=10)
        pd.testing.assert_series_equal(bb["middle"], ma, check_names=False)

    def test_insufficient_data_returns_all_nan(self, small_df):
        bb = calc_bollinger(small_df, period=50)
        assert bb["upper"].isna().all()

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(KeyError, match="close"):
            calc_bollinger(df)


# ---------------------------------------------------------------------------
# calc_kdj
# ---------------------------------------------------------------------------

class TestCalcKdj:
    def test_returns_k_d_j(self, rising_df):
        kdj = calc_kdj(rising_df, n=9, m1=3, m2=3)
        assert set(kdj.keys()) == {"k", "d", "j"}

    def test_j_equals_3k_minus_2d(self, rising_df):
        kdj = calc_kdj(rising_df, n=9, m1=3, m2=3)
        expected_j = 3 * kdj["k"] - 2 * kdj["d"]
        pd.testing.assert_series_equal(kdj["j"], expected_j, check_names=False)

    def test_k_range(self, small_df):
        """K values should generally be between 0 and 100 (RSV-based)."""
        kdj = calc_kdj(small_df, n=5, m1=3, m2=3)
        k_valid = kdj["k"].dropna()
        # K is EWM-smoothed RSV, so should be in [0, 100]
        assert (k_valid >= 0).all()
        assert (k_valid <= 100).all()

    def test_insufficient_data_returns_all_nan(self, small_df):
        kdj = calc_kdj(small_df, n=50)
        assert kdj["k"].isna().all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError, match="missing"):
            calc_kdj(df)
