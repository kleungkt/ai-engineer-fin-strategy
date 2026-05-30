"""Tests for the patterns module (patterns.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patterns import (
    detect_doji,
    detect_engulfing,
    detect_evening_star,
    detect_hammer,
    detect_morning_star,
    detect_three_black_crows,
    detect_three_white_soldiers,
    get_pattern_summary,
    scan_patterns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of row dicts.

    Each dict should have keys: open, high, low, close.
    date and volume are auto-generated.
    """
    n = len(rows)
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = {
        "date": dates,
        "open": [r["open"] for r in rows],
        "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r.get("volume", 1_000_000) for r in rows],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# detect_doji
# ---------------------------------------------------------------------------

class TestDetectDoji:
    def test_small_body_is_doji(self):
        """Tiny body relative to range → doji."""
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.05},  # body=0.05, range=10
        ])
        result = detect_doji(df)
        assert result.iloc[0] is True or result.iloc[0] == True

    def test_large_body_not_doji(self):
        """Large body relative to range → not doji."""
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 104.0},  # body=4, range=10
        ])
        result = detect_doji(df)
        assert result.iloc[0] == False

    def test_zero_range_not_doji(self):
        """Zero range (flat candle) → division-by-zero guard, should be False."""
        df = _make_df([
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ])
        result = detect_doji(df)
        assert result.iloc[0] == False

    def test_custom_threshold(self):
        """With threshold=0.5, a body/range of 0.3 should still be doji."""
        df = _make_df([
            {"open": 100.0, "high": 110.0, "low": 90.0, "close": 103.0},  # body=3, range=20
        ])
        result = detect_doji(df, threshold=0.5)
        assert result.iloc[0] == True

    def test_empty_df(self):
        df = _make_df([])
        result = detect_doji(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# detect_engulfing
# ---------------------------------------------------------------------------

class TestDetectEngulfing:
    def test_bullish_engulfing(self):
        """Prev bearish (open>close), current bullish with body engulfing prev."""
        df = _make_df([
            {"open": 105.0, "high": 106.0, "low": 98.0, "close": 100.0},   # bearish
            {"open": 99.0, "high": 107.0, "low": 98.0, "close": 106.0},    # bullish engulfs
        ])
        result = detect_engulfing(df)
        assert result.iloc[0] == False  # first row always False
        assert result.iloc[1] == True

    def test_bearish_engulfing(self):
        """Prev bullish (close>open), current bearish with body engulfing prev."""
        df = _make_df([
            {"open": 100.0, "high": 106.0, "low": 99.0, "close": 105.0},   # bullish
            {"open": 106.0, "high": 107.0, "low": 99.0, "close": 99.0},    # bearish engulfs
        ])
        result = detect_engulfing(df)
        assert result.iloc[0] == False
        assert result.iloc[1] == True

    def test_no_engulfing(self):
        """Consecutive small candles with no engulfing."""
        df = _make_df([
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"open": 100.5, "high": 101.0, "low": 99.5, "close": 100.8},
        ])
        result = detect_engulfing(df)
        assert result.sum() == 0

    def test_single_row(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 103.0},
        ])
        result = detect_engulfing(df)
        assert result.iloc[0] == False


# ---------------------------------------------------------------------------
# detect_hammer
# ---------------------------------------------------------------------------

class TestDetectHammer:
    def test_hammer_long_lower_shadow(self):
        """Small body at top, long lower shadow (>= 2x body)."""
        # open=100, close=101 → body=1, low=95 → lower shadow=5, high=102 → upper=1
        df = _make_df([
            {"open": 100.0, "high": 102.0, "low": 95.0, "close": 101.0},
        ])
        result = detect_hammer(df)
        assert result.iloc[0] == True

    def test_no_hammer_large_body(self):
        """Large body relative to shadows → not hammer."""
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 104.0},
        ])
        result = detect_hammer(df)
        assert result.iloc[0] == False

    def test_inverted_hammer(self):
        """Long upper shadow, small body, small lower shadow."""
        # open=100, close=101 → body=1, high=107 → upper=6, low=99.5 → lower=0.5
        df = _make_df([
            {"open": 100.0, "high": 107.0, "low": 99.5, "close": 101.0},
        ])
        result = detect_hammer(df)
        assert result.iloc[0] == True

    def test_zero_range_not_hammer(self):
        df = _make_df([
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ])
        result = detect_hammer(df)
        assert result.iloc[0] == False


# ---------------------------------------------------------------------------
# detect_morning_star / evening_star
# ---------------------------------------------------------------------------

class TestMorningStar:
    def test_morning_star_pattern(self):
        """Bearish candle → small gap-down star → bullish recovery."""
        df = _make_df([
            {"open": 105.0, "high": 106.0, "low": 97.0, "close": 98.0},    # bearish
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5},      # small star
            {"open": 96.0, "high": 103.0, "low": 95.5, "close": 102.0},    # bullish, closes above mid of 1st
        ])
        result = detect_morning_star(df)
        assert result.iloc[2] == True

    def test_morning_star_insufficient_data(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 98.0},
        ])
        result = detect_morning_star(df)
        assert result.sum() == 0


class TestEveningStar:
    def test_evening_star_pattern(self):
        """Bullish candle → small gap-up star → bearish drop."""
        df = _make_df([
            {"open": 100.0, "high": 107.0, "low": 99.0, "close": 106.0},   # bullish
            {"open": 108.0, "high": 109.0, "low": 107.0, "close": 108.5},  # small star
            {"open": 107.0, "high": 108.0, "low": 100.0, "close": 101.0},  # bearish, below mid of 1st
        ])
        result = detect_evening_star(df)
        assert result.iloc[2] == True

    def test_evening_star_insufficient_data(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 103.0},
        ])
        result = detect_evening_star(df)
        assert result.sum() == 0


# ---------------------------------------------------------------------------
# detect_three_white_soldiers / three_black_crows
# ---------------------------------------------------------------------------

class TestThreeWhiteSoldiers:
    def test_three_white_soldiers(self):
        """Three bullish candles, each opening within prior body, closing higher."""
        df = _make_df([
            {"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
            {"open": 101.0, "high": 107.0, "low": 100.5, "close": 106.0},
            {"open": 104.0, "high": 110.0, "low": 103.0, "close": 109.0},
        ])
        result = detect_three_white_soldiers(df)
        assert result.iloc[2] == True

    def test_not_white_soldiers_bearish(self):
        """If any candle is bearish, pattern fails."""
        df = _make_df([
            {"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
            {"open": 102.0, "high": 106.0, "low": 101.0, "close": 105.0},
            {"open": 106.0, "high": 107.0, "low": 100.0, "close": 101.0},  # bearish
        ])
        result = detect_three_white_soldiers(df)
        assert result.iloc[2] == False


class TestThreeBlackCrows:
    def test_three_black_crows(self):
        """Three bearish candles, each opening within prior body, closing lower."""
        df = _make_df([
            {"open": 110.0, "high": 111.0, "low": 103.0, "close": 104.0},
            {"open": 106.0, "high": 107.0, "low": 100.0, "close": 101.0},
            {"open": 103.0, "high": 104.0, "low": 96.0, "close": 97.0},
        ])
        result = detect_three_black_crows(df)
        assert result.iloc[2] == True

    def test_not_black_crows_bullish(self):
        df = _make_df([
            {"open": 110.0, "high": 111.0, "low": 103.0, "close": 104.0},
            {"open": 106.0, "high": 107.0, "low": 100.0, "close": 101.0},
            {"open": 100.0, "high": 108.0, "low": 99.0, "close": 107.0},  # bullish
        ])
        result = detect_three_black_crows(df)
        assert result.iloc[2] == False


# ---------------------------------------------------------------------------
# scan_patterns
# ---------------------------------------------------------------------------

class TestScanPatterns:
    def test_returns_all_pattern_names(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
        ])
        result = scan_patterns(df)
        expected = {
            "engulfing", "doji", "hammer", "morning_star",
            "evening_star", "three_white_soldiers", "three_black_crows",
        }
        assert set(result.keys()) == expected

    def test_each_value_is_series(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
            {"open": 101.0, "high": 106.0, "low": 100.0, "close": 105.0},
        ])
        result = scan_patterns(df)
        for name, series in result.items():
            assert isinstance(series, pd.Series), f"{name} is not a Series"
            assert len(series) == 2


# ---------------------------------------------------------------------------
# get_pattern_summary
# ---------------------------------------------------------------------------

class TestGetPatternSummary:
    def test_returns_string(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
        ])
        result = get_pattern_summary(df)
        assert isinstance(result, str)

    def test_contains_header(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
        ])
        result = get_pattern_summary(df)
        assert "K-Line Pattern Summary" in result

    def test_contains_pattern_names(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
        ])
        result = get_pattern_summary(df)
        assert "Engulfing" in result
        assert "Doji" in result
        assert "Hammer" in result

    def test_contains_data_points(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0},
            {"open": 101.0, "high": 106.0, "low": 100.0, "close": 105.0},
        ])
        result = get_pattern_summary(df)
        assert "2" in result  # data points analyzed


# ---------------------------------------------------------------------------
# Edge cases: insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_all_detectors_handle_empty_df(self):
        """All detectors should return empty Series for empty input."""
        df = _make_df([])
        assert len(detect_doji(df)) == 0
        assert len(detect_engulfing(df)) == 0
        assert len(detect_hammer(df)) == 0
        assert len(detect_morning_star(df)) == 0
        assert len(detect_evening_star(df)) == 0
        assert len(detect_three_white_soldiers(df)) == 0
        assert len(detect_three_black_crows(df)) == 0

    def test_single_row_detectors(self):
        """Single-candle detectors should handle 1 row."""
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
        ])
        # These should return Series of length 1
        assert len(detect_doji(df)) == 1
        assert len(detect_hammer(df)) == 1
        # Engulfing needs 2 rows but should still handle 1
        assert len(detect_engulfing(df)) == 1
        assert detect_engulfing(df).iloc[0] == False
