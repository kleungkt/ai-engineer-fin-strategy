"""Tests for the data_fetcher module (data_fetcher.py).

All akshare calls are mocked — no real network requests are made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import data_fetcher
from data_fetcher import (
    fetch_index_data,
    fetch_stock_daily,
    fetch_stock_list,
    _get_cached,
    _set_cached,
    _retry,
    _cache,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sample_raw_df(days: int = 10) -> pd.DataFrame:
    """Simulate akshare's Chinese-column raw DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": [100.0 + i for i in range(days)],
            "最高": [105.0 + i for i in range(days)],
            "最低": [95.0 + i for i in range(days)],
            "收盘": [102.0 + i for i in range(days)],
            "成交量": [1_000_000 + i * 10_000 for i in range(days)],
        }
    )


def _sample_index_df(days: int = 10) -> pd.DataFrame:
    """Simulate akshare's English-column index DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [3000.0 + i for i in range(days)],
            "high": [3050.0 + i for i in range(days)],
            "low": [2950.0 + i for i in range(days)],
            "close": [3020.0 + i for i in range(days)],
            "volume": [10_000_000 + i * 100_000 for i in range(days)],
        }
    )


def _sample_stock_list_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["600519", "000858", "601318"],
            "name": ["贵州茅台", "五粮液", "中国平安"],
        }
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:
    def test_get_cached_miss(self):
        assert _get_cached("nonexistent") is None

    def test_set_and_get(self):
        _set_cached("key1", "value1")
        assert _get_cached("key1") == "value1"

    def test_ttl_expiry(self):
        _set_cached("key2", "value2")
        # Manually backdate the timestamp
        _cache["key2"] = (time.time() - 10_000, "value2")
        assert _get_cached("key2", ttl=3600) is None

    def test_stale_key_deleted(self):
        _set_cached("key3", "value3")
        _cache["key3"] = (time.time() - 10_000, "value3")
        _get_cached("key3", ttl=3600)
        assert "key3" not in _cache

    def test_fresh_key_within_ttl(self):
        _set_cached("key4", "value4")
        assert _get_cached("key4", ttl=3600) == "value4"


# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_success_first_try(self):
        fn = MagicMock(return_value="ok")
        result = _retry(fn, retries=3, delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    def test_success_after_failures(self):
        fn = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "ok"])
        result = _retry(fn, retries=3, delay=0.01)
        assert result == "ok"
        assert fn.call_count == 3

    def test_all_retries_fail(self):
        fn = MagicMock(side_effect=Exception("always fail"))
        with pytest.raises(RuntimeError, match="All 3 attempts failed"):
            _retry(fn, retries=3, delay=0.01)
        assert fn.call_count == 3

    def test_single_retry_success(self):
        fn = MagicMock(return_value=42)
        result = _retry(fn, retries=1, delay=0.01)
        assert result == 42


# ---------------------------------------------------------------------------
# fetch_stock_daily
# ---------------------------------------------------------------------------

class TestFetchStockDaily:
    @patch("data_fetcher.ak")
    def test_column_normalization(self, mock_ak):
        """Chinese column names should be renamed to English."""
        mock_ak.stock_zh_a_hist.return_value = _sample_raw_df(10)
        df = fetch_stock_daily("600519", days=10)

        expected_cols = {"date", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_cols

    @patch("data_fetcher.ak")
    def test_returns_requested_days(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = _sample_raw_df(10)
        df = fetch_stock_daily("600519", days=5)
        assert len(df) == 5

    @patch("data_fetcher.ak")
    def test_date_is_datetime(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = _sample_raw_df(10)
        df = fetch_stock_daily("600519", days=10)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    @patch("data_fetcher.ak")
    def test_caching(self, mock_ak):
        """Second call should use cache, not call akshare again."""
        mock_ak.stock_zh_a_hist.return_value = _sample_raw_df(10)

        df1 = fetch_stock_daily("600519", days=10)
        df2 = fetch_stock_daily("600519", days=10)

        assert df1.equals(df2)
        mock_ak.stock_zh_a_hist.assert_called_once()

    @patch("data_fetcher.ak")
    def test_different_symbols_different_cache(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = _sample_raw_df(10)

        fetch_stock_daily("600519", days=10)
        fetch_stock_daily("000858", days=10)

        assert mock_ak.stock_zh_a_hist.call_count == 2

    @patch("data_fetcher.ak")
    def test_retry_on_failure(self, mock_ak):
        """Should retry on exception."""
        mock_ak.stock_zh_a_hist.side_effect = [
            Exception("network error"),
            _sample_raw_df(10),
        ]
        df = fetch_stock_daily("600519", days=10)
        assert len(df) == 10
        assert mock_ak.stock_zh_a_hist.call_count == 2

    @patch("data_fetcher.ak")
    def test_all_retries_fail_raises(self, mock_ak):
        mock_ak.stock_zh_a_hist.side_effect = Exception("permanent failure")
        with pytest.raises(RuntimeError, match="attempts failed"):
            fetch_stock_daily("600519", days=10)


# ---------------------------------------------------------------------------
# fetch_stock_list
# ---------------------------------------------------------------------------

class TestFetchStockList:
    @patch("data_fetcher.ak")
    def test_returns_list_of_dicts(self, mock_ak):
        mock_ak.stock_info_a_code_name.return_value = _sample_stock_list_df()
        result = fetch_stock_list()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    @patch("data_fetcher.ak")
    def test_has_symbol_and_name(self, mock_ak):
        mock_ak.stock_info_a_code_name.return_value = _sample_stock_list_df()
        result = fetch_stock_list()
        for r in result:
            assert "symbol" in r
            assert "name" in r

    @patch("data_fetcher.ak")
    def test_column_rename(self, mock_ak):
        """'code' column should become 'symbol'."""
        mock_ak.stock_info_a_code_name.return_value = _sample_stock_list_df()
        result = fetch_stock_list()
        symbols = {r["symbol"] for r in result}
        assert "600519" in symbols

    @patch("data_fetcher.ak")
    def test_caching(self, mock_ak):
        mock_ak.stock_info_a_code_name.return_value = _sample_stock_list_df()
        fetch_stock_list()
        fetch_stock_list()
        mock_ak.stock_info_a_code_name.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_index_data
# ---------------------------------------------------------------------------

class TestFetchIndexData:
    @patch("data_fetcher.ak")
    def test_returns_correct_columns(self, mock_ak):
        mock_ak.stock_zh_index_daily.return_value = _sample_index_df(10)
        df = fetch_index_data("sh000001", days=10)
        expected = {"date", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected

    @patch("data_fetcher.ak")
    def test_returns_requested_days(self, mock_ak):
        mock_ak.stock_zh_index_daily.return_value = _sample_index_df(20)
        df = fetch_index_data("sh000001", days=5)
        assert len(df) == 5

    @patch("data_fetcher.ak")
    def test_caching(self, mock_ak):
        mock_ak.stock_zh_index_daily.return_value = _sample_index_df(10)
        fetch_index_data("sh000001", days=10)
        fetch_index_data("sh000001", days=10)
        mock_ak.stock_zh_index_daily.assert_called_once()

    @patch("data_fetcher.ak")
    def test_date_is_datetime(self, mock_ak):
        mock_ak.stock_zh_index_daily.return_value = _sample_index_df(10)
        df = fetch_index_data("sh000001", days=10)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    @patch("data_fetcher.ak")
    def test_retry_on_failure(self, mock_ak):
        mock_ak.stock_zh_index_daily.side_effect = [
            Exception("timeout"),
            _sample_index_df(10),
        ]
        df = fetch_index_data("sh000001", days=10)
        assert len(df) == 10
        assert mock_ak.stock_zh_index_daily.call_count == 2
