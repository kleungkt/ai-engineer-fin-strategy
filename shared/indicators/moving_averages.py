"""
Moving Averages Module
======================

This module implements various moving average calculations commonly used
in technical analysis for stock market data.

Supported indicators:
- SMA: Simple Moving Average
- EMA: Exponential Moving Average
- WMA: Weighted Moving Average

All functions accept pandas Series or DataFrames and return pandas Series.
"""

import pandas as pd
import numpy as np
from typing import Union


def calculate_sma(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 20,
    column: str = "close"
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        data: Price data as pandas Series or DataFrame
        period: Number of periods for the moving average (default: 20)
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        pd.Series: SMA values

    Example:
        >>> sma_20 = calculate_sma(df, period=20)
        >>> sma_50 = calculate_sma(df, period=50)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算简单移动平均线
    return series.rolling(window=period, min_periods=1).mean()


def calculate_ema(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 20,
    column: str = "close",
    adjust: bool = False
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        data: Price data as pandas Series or DataFrame
        period: Number of periods for the moving average (default: 20)
        column: Column name to use if data is a DataFrame (default: 'close')
        adjust: Whether to adjust for bias (default: False)

    Returns:
        pd.Series: EMA values

    Example:
        >>> ema_12 = calculate_ema(df, period=12)
        >>> ema_26 = calculate_ema(df, period=26)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算指数移动平均线
    return series.ewm(span=period, adjust=adjust).mean()


def calculate_wma(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 20,
    column: str = "close"
) -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA).

    Args:
        data: Price data as pandas Series or DataFrame
        period: Number of periods for the moving average (default: 20)
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        pd.Series: WMA values

    Example:
        >>> wma_20 = calculate_wma(df, period=20)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算权重: [1, 2, 3, ..., period]
    weights = np.arange(1, period + 1)

    # 计算加权移动平均线
    def weighted_mean(x):
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period).apply(weighted_mean, raw=True)


def calculate_crossover(
    fast_ma: pd.Series,
    slow_ma: pd.Series
) -> pd.Series:
    """
    Detect crossover between two moving averages.

    Args:
        fast_ma: Faster moving average series
        slow_ma: Slower moving average series

    Returns:
        pd.Series: Boolean series indicating crossover points
                   (True when fast_ma crosses above slow_ma)

    Example:
        >>> sma_20 = calculate_sma(df, 20)
        >>> sma_50 = calculate_sma(df, 50)
        >>> golden_cross = calculate_crossover(sma_20, sma_50)
    """
    # 当前时刻快线在慢线上方
    above = fast_ma > slow_ma
    # 前一时刻快线在慢线下方
    prev_below = fast_ma.shift(1) <= slow_ma.shift(1)
    # 金叉: 从下方穿越到上方
    return above & prev_below
