"""
Technical indicators module for stock analysis.

Provides calculation functions for common technical indicators (MA, EMA, MACD,
RSI, Bollinger Bands, KDJ) and condition checker helpers for crossover/crossunder
and threshold detection.

All functions expect a pandas DataFrame with columns:
    date, open, high, low, close, volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator calculation functions
# ---------------------------------------------------------------------------


def calc_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average (SMA) of the closing price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    period : int
        Look-back window length (default 20).

    Returns
    -------
    pd.Series
        SMA values aligned to the original index.  The first ``period - 1``
        entries are NaN.
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column")
    if len(df) < period:
        # Not enough data – return all-NaN series
        return pd.Series(np.nan, index=df.index, name=f"ma_{period}")
    return df["close"].rolling(window=period, min_periods=period).mean()


def calc_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average (EMA) of the closing price.

    Uses the standard pandas ``ewm`` implementation with ``adjust=False``
    so that the first valid value seeds the EMA.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    period : int
        Look-back window length (default 20).

    Returns
    -------
    pd.Series
        EMA values aligned to the original index.
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column")
    if len(df) < period:
        return pd.Series(np.nan, index=df.index, name=f"ema_{period}")
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence / Divergence).

    The MACD line is the difference between the *fast* EMA and the *slow* EMA
    of the closing price.  The signal line is the EMA of the MACD line.  The
    histogram is ``macd - signal``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal-line EMA period (default 9).

    Returns
    -------
    dict[str, pd.Series]
        ``{"macd": <macd line>, "signal": <signal line>, "histogram": <histogram>}``
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column")
    if len(df) < slow:
        nan = pd.Series(np.nan, index=df.index)
        return {"macd": nan.copy(), "signal": nan.copy(), "histogram": nan.copy()}

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": macd_line.rename("macd"),
        "signal": signal_line.rename("macd_signal"),
        "histogram": histogram.rename("macd_histogram"),
    }


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI oscillates between 0 and 100.  Values above 70 are traditionally
    considered overbought, and below 30 oversold.

    The implementation uses Wilder's smoothing (EWM with ``adjust=False``
    and ``alpha = 1/period``).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values (0-100) aligned to the original index.
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column")
    if len(df) < period + 1:
        return pd.Series(np.nan, index=df.index, name="rsi")

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (equivalent to alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "rsi"
    return rsi


def calc_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: int = 2,
) -> dict[str, pd.Series]:
    """Calculate Bollinger Bands.

    Bands are constructed around a simple moving average of the closing price:

    * **middle** = SMA(close, period)
    * **upper**  = middle + std × rolling_std(close, period)
    * **lower**  = middle - std × rolling_std(close, period)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    period : int
        Look-back window (default 20).
    std : int | float
        Number of standard deviations for the bands (default 2).

    Returns
    -------
    dict[str, pd.Series]
        ``{"upper": ..., "middle": ..., "lower": ...}``
    """
    if "close" not in df.columns:
        raise KeyError("DataFrame must contain a 'close' column")
    if len(df) < period:
        nan = pd.Series(np.nan, index=df.index)
        return {"upper": nan.copy(), "middle": nan.copy(), "lower": nan.copy()}

    middle = df["close"].rolling(window=period, min_periods=period).mean()
    rolling_std = df["close"].rolling(window=period, min_periods=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std

    return {
        "upper": upper.rename("bb_upper"),
        "middle": middle.rename("bb_middle"),
        "lower": lower.rename("bb_lower"),
    }


def calc_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> dict[str, pd.Series]:
    """Calculate KDJ (Stochastic Oscillator variant popular in Asian markets).

    Formulas
    --------
    RSV = (close - lowest_low_n) / (highest_high_n - lowest_low_n) × 100

    * K = SMA of RSV with multiplier *m1*  (implemented as EWM-style smoothing)
    * D = SMA of K  with multiplier *m2*
    * J = 3 × K − 2 × D

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``high``, ``low``, and ``close`` columns.
    n : int
        Look-back window for RSV (default 9).
    m1 : int
        Smoothing period for K (default 3).
    m2 : int
        Smoothing period for D (default 3).

    Returns
    -------
    dict[str, pd.Series]
        ``{"k": ..., "d": ..., "j": ...}``
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")
    if len(df) < n:
        nan = pd.Series(np.nan, index=df.index)
        return {"k": nan.copy(), "d": nan.copy(), "j": nan.copy()}

    lowest_low = df["low"].rolling(window=n, min_periods=n).min()
    highest_high = df["high"].rolling(window=n, min_periods=n).max()

    # RSV (Raw Stochastic Value)
    denominator = highest_high - lowest_low
    # Avoid division by zero when high == low over the window
    rsv = np.where(
        denominator == 0,
        50.0,  # neutral value when there is no range
        (df["close"] - lowest_low) / denominator * 100,
    )
    rsv = pd.Series(rsv, index=df.index, name="rsv")

    # K and D use a recursive smoothing similar to EWM with alpha = 1/m
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "k": k.rename("kdj_k"),
        "d": d.rename("kdj_d"),
        "j": j.rename("kdj_j"),
    }


# ---------------------------------------------------------------------------
# Condition checker helpers
# ---------------------------------------------------------------------------


def check_crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Return a boolean Series that is ``True`` where *series_a* crosses **above**
    *series_b*.

    A crossover at index *i* means ``series_a[i] > series_b[i]`` **and**
    ``series_a[i-1] <= series_b[i-1]``.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Numeric series of equal length and index alignment.

    Returns
    -------
    pd.Series (bool)
    """
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))


def check_crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Return a boolean Series that is ``True`` where *series_a* crosses **below**
    *series_b*.

    A crossunder at index *i* means ``series_a[i] < series_b[i]`` **and**
    ``series_a[i-1] >= series_b[i-1]``.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Numeric series of equal length and index alignment.

    Returns
    -------
    pd.Series (bool)
    """
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))


def check_above(series: pd.Series, value: float) -> pd.Series:
    """Return a boolean Series that is ``True`` where *series* > *value*.

    Parameters
    ----------
    series : pd.Series
        Numeric series to test.
    value : float
        Threshold value.

    Returns
    -------
    pd.Series (bool)
    """
    return series > value


def check_below(series: pd.Series, value: float) -> pd.Series:
    """Return a boolean Series that is ``True`` where *series* < *value*.

    Parameters
    ----------
    series : pd.Series
        Numeric series to test.
    value : float
        Threshold value.

    Returns
    -------
    pd.Series (bool)
    """
    return series < value
