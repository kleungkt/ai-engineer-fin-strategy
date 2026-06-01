"""Tests for the data_fetcher module."""

import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data_fetcher import generate_sample_data, fetch_stock_daily


class TestGenerateSampleData:
    """Tests for generate_sample_data()."""

    def test_returns_dataframe(self):
        df = generate_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_default_columns(self):
        df = generate_sample_data()
        expected_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_default_days(self):
        df = generate_sample_data()
        assert len(df) == 500

    def test_custom_days(self):
        df = generate_sample_data(days=100)
        assert len(df) == 100

    def test_custom_symbol(self):
        df = generate_sample_data(symbol="TEST")
        assert len(df) == 500  # symbol doesn't affect size

    def test_prices_are_positive(self):
        df = generate_sample_data(days=200)
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"Non-positive values in {col}"

    def test_volume_is_positive(self):
        df = generate_sample_data(days=200)
        assert (df["volume"] > 0).all()

    def test_high_ge_low(self):
        df = generate_sample_data(days=200)
        assert (df["high"] >= df["low"]).all()

    def test_dates_are_sorted(self):
        df = generate_sample_data(days=200)
        dates = df["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_deterministic_with_seed(self):
        """generate_sample_data uses np.random.seed(42), so results should be identical."""
        df1 = generate_sample_data(days=50)
        df2 = generate_sample_data(days=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_date_column_is_datetime(self):
        df = generate_sample_data(days=50)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_numeric_columns(self):
        df = generate_sample_data(days=50)
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_numeric_dtype(df[col])


def _make_mock_akshare(n_rows=100):
    """Build a mock akshare module with stock_zh_a_hist returning Chinese-column data."""
    mock_ak = MagicMock()
    mock_df = pd.DataFrame({
        "日期": pd.bdate_range("2024-01-01", periods=n_rows),
        "开盘": np.random.uniform(90, 110, n_rows),
        "最高": np.random.uniform(100, 115, n_rows),
        "最低": np.random.uniform(85, 100, n_rows),
        "收盘": np.random.uniform(90, 110, n_rows),
        "成交量": np.random.uniform(1e6, 5e6, n_rows),
    })
    mock_ak.stock_zh_a_hist.return_value = mock_df
    return mock_ak, mock_df


class TestFetchStockDaily:
    """Tests for fetch_stock_daily() with mocked akshare."""

    def test_fetch_a_share_success(self):
        """Test fetching A-share data with mocked akshare."""
        np.random.seed(99)
        mock_ak, mock_df = _make_mock_akshare(100)
        sys.modules["akshare"] = mock_ak
        try:
            df = fetch_stock_daily("000001", days=50)
            assert isinstance(df, pd.DataFrame)
            expected_cols = ["date", "open", "high", "low", "close", "volume"]
            for col in expected_cols:
                assert col in df.columns
            assert len(df) <= 50
            mock_ak.stock_zh_a_hist.assert_called_once()
        finally:
            del sys.modules["akshare"]

    def test_fetch_without_akshare_raises(self):
        """When akshare is not installed, should raise ImportError."""
        # Temporarily remove akshare from sys.modules if present
        saved = sys.modules.pop("akshare", None)
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "akshare":
                raise ImportError("No module named 'akshare'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="akshare"):
                fetch_stock_daily("000001")
        finally:
            builtins.__import__ = original_import
            if saved is not None:
                sys.modules["akshare"] = saved

    def test_fetch_normalizes_columns(self):
        """Test that Chinese column names get normalized to English."""
        np.random.seed(99)
        mock_ak, _ = _make_mock_akshare(60)
        sys.modules["akshare"] = mock_ak
        try:
            df = fetch_stock_daily("000001", days=50)
            assert "date" in df.columns
            assert "open" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns
            # Should not have Chinese column names
            assert "日期" not in df.columns
        finally:
            del sys.modules["akshare"]

    def test_fetch_returns_limited_days(self):
        """Result should be limited to the requested number of days."""
        np.random.seed(99)
        mock_ak, _ = _make_mock_akshare(200)
        sys.modules["akshare"] = mock_ak
        try:
            df = fetch_stock_daily("000001", days=50)
            assert len(df) <= 50
        finally:
            del sys.modules["akshare"]

    def test_fetch_returns_numeric_types(self):
        """Returned price columns should be float."""
        np.random.seed(99)
        mock_ak, _ = _make_mock_akshare(60)
        sys.modules["akshare"] = mock_ak
        try:
            df = fetch_stock_daily("000001", days=50)
            for col in ["open", "high", "low", "close"]:
                assert df[col].dtype == float
        finally:
            del sys.modules["akshare"]

    def test_fetch_with_failing_a_share_falls_back_to_us(self):
        """If stock_zh_a_hist fails, it should try stock_us_daily."""
        np.random.seed(99)
        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.side_effect = ValueError("not found")

        us_df = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=60),
            "Open": [100.0] * 60,
            "High": [105.0] * 60,
            "Low": [95.0] * 60,
            "Close": [102.0] * 60,
            "Volume": [1e6] * 60,
        })
        mock_ak.stock_us_daily.return_value = us_df

        sys.modules["akshare"] = mock_ak
        try:
            df = fetch_stock_daily("AAPL", days=50)
            assert isinstance(df, pd.DataFrame)
            assert "date" in df.columns
            mock_ak.stock_us_daily.assert_called_once()
        finally:
            del sys.modules["akshare"]
