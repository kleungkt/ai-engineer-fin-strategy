"""
Oscillators Module
==================

This module implements various oscillator indicators commonly used
in technical analysis for stock market data.

Supported indicators:
- MACD: Moving Average Convergence Divergence
- RSI: Relative Strength Index
- Stochastic: Stochastic Oscillator

All functions accept pandas Series or DataFrames and return pandas Series or DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def calculate_macd(
    data: Union[pd.Series, pd.DataFrame],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "close"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Price data as pandas Series or DataFrame
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line (default: 9)
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        Tuple containing:
        - macd_line: MACD line (fast EMA - slow EMA)
        - signal_line: Signal line (EMA of MACD line)
        - histogram: MACD histogram (MACD line - signal line)

    Example:
        >>> macd, signal, hist = calculate_macd(df)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算快速 EMA 和慢速 EMA
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()

    # MACD 线 = 快速 EMA - 慢速 EMA
    macd_line = fast_ema - slow_ema

    # 信号线 = MACD 线的 EMA
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # 柱状图 = MACD 线 - 信号线
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_rsi(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 14,
    column: str = "close"
) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        data: Price data as pandas Series or DataFrame
        period: Number of periods for RSI calculation (default: 14)
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        pd.Series: RSI values (0-100)

    Example:
        >>> rsi = calculate_rsi(df, period=14)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算价格变化
    delta = series.diff()

    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # 计算平均上涨和平均下跌 (使用 EMA 平滑)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # 计算相对强度 RS
    rs = avg_gain / avg_loss

    # 计算 RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Period for %K line (default: 14)
        d_period: Period for %D line (default: 3)

    Returns:
        Tuple containing:
        - k_line: %K line
        - d_line: %D line (moving average of %K)

    Example:
        >>> k, d = calculate_stochastic(df['high'], df['low'], df['close'])
    """
    # 计算最低低点和最高高点
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    # 计算 %K 线
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # 计算 %D 线 (%K 的移动平均)
    d_line = k_line.rolling(window=d_period).mean()

    return k_line, d_line


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Look-back period (default: 14)

    Returns:
        pd.Series: Williams %R values (-100 to 0)

    Example:
        >>> williams = calculate_williams_r(df['high'], df['low'], df['close'])
    """
    # 计算最高高点和最低低点
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    # 计算 Williams %R
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)

    return williams_r
