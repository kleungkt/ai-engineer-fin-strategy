"""
Data fetcher module for AI Strategy Generator.

Provides stock data fetching via akshare and sample data generation for testing.
"""

import datetime
import numpy as np
import pandas as pd


def fetch_stock_daily(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily stock data using akshare.

    Args:
        symbol: Stock symbol (e.g., '000001' for Ping An Bank)
        days: Number of trading days of history to fetch

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    try:
        import akshare as ak

        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

        # Try A-share first
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
        except Exception:
            # Try US stock via akshare
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
            except Exception:
                raise ValueError(f"Unable to fetch data for symbol: {symbol}")

        # Normalize column names
        column_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Ensure required columns exist
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                # Try lowercase
                if col.lower() in [c.lower() for c in df.columns]:
                    actual = [c for c in df.columns if c.lower() == col.lower()][0]
                    df = df.rename(columns={actual: col})
                else:
                    raise ValueError(f"Column '{col}' not found in fetched data")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days).reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float)

        return df[required]

    except ImportError:
        raise ImportError("akshare is not installed. Install it with: pip install akshare")


def generate_sample_data(symbol: str = "SAMPLE", days: int = 500) -> pd.DataFrame:
    """
    Generate synthetic stock data for testing purposes.

    Args:
        symbol: Symbol label for the generated data
        days: Number of trading days to generate

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(42)

    dates = pd.bdate_range(end=datetime.date.today(), periods=days)

    # Generate price series using geometric Brownian motion
    initial_price = 100.0
    drift = 0.0002
    volatility = 0.02

    returns = np.random.normal(drift, volatility, days)
    prices = initial_price * np.cumprod(1 + returns)

    # Build OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
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

    df = pd.DataFrame(data)
    return df
