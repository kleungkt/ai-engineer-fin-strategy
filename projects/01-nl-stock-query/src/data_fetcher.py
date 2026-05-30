"""
Real stock data fetching module using akshare.
Fetches China A-stock daily OHLCV data, stock lists, and index data.
Includes in-memory caching with TTL and retry logic.
"""

import logging
import time
from typing import Optional
from functools import lru_cache

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

# In-memory cache: key -> (timestamp, data)
_cache: dict[str, tuple[float, object]] = {}
DEFAULT_TTL = 3600  # 1 hour in seconds


def _get_cached(key: str, ttl: int = DEFAULT_TTL) -> Optional[object]:
    """Return cached value if fresh, else None."""
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
        del _cache[key]
    return None


def _set_cached(key: str, val: object) -> None:
    _cache[key] = (time.time(), val)


def _retry(fn, retries: int = 3, delay: float = 2.0):
    """Retry a callable up to `retries` times with exponential backoff."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"All {retries} attempts failed") from last_exc


def fetch_stock_daily(symbol: str, days: int = 120) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single A-stock symbol.

    Args:
        symbol: A-share stock code, e.g. '600519' for 贵州茅台.
        days: Number of recent trading days to return.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    cache_key = f"stock_daily:{symbol}:{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    def _fetch() -> pd.DataFrame:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            adjust="qfq",
        )
        # Normalize column names
        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=col_map)
        wanted = ["date", "open", "high", "low", "close", "volume"]
        df = df[wanted].tail(days).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    logger.info("Fetching daily data for %s (last %d days)", symbol, days)
    df = _retry(_fetch)
    _set_cached(cache_key, df)
    return df


def fetch_stock_list(market: str = "A股") -> list[dict]:
    """Fetch list of stocks with basic info.

    Args:
        market: Market identifier (currently only 'A股' supported).

    Returns:
        List of dicts with keys: symbol, name.
    """
    cache_key = f"stock_list:{market}"
    cached = _get_cached(cache_key, ttl=86400)  # cache list for 24h
    if cached is not None:
        return cached  # type: ignore[return-value]

    def _fetch() -> list[dict]:
        df = ak.stock_info_a_code_name()
        col_map = {"code": "symbol", "name": "name"}
        df = df.rename(columns=col_map)
        records = df[["symbol", "name"]].to_dict(orient="records")
        return records

    logger.info("Fetching stock list for market=%s", market)
    result = _retry(_fetch)
    _set_cached(cache_key, result)
    return result


def fetch_index_data(symbol: str = "sh000001", days: int = 120) -> pd.DataFrame:
    """Fetch index daily data (e.g. 上证指数).

    Args:
        symbol: Index code, e.g. 'sh000001' for 上证指数.
        days: Number of recent trading days to return.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    cache_key = f"index:{symbol}:{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    def _fetch() -> pd.DataFrame:
        df = ak.stock_zh_index_daily(symbol=symbol)
        col_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        # Column names may already be English; normalize just in case
        df = df.rename(columns=col_map)
        wanted = ["date", "open", "high", "low", "close", "volume"]
        df = df[wanted].tail(days).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    logger.info("Fetching index data for %s (last %d days)", symbol, days)
    df = _retry(_fetch)
    _set_cached(cache_key, df)
    return df
