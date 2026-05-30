"""
Bollinger Bands Module
======================

This module implements Bollinger Bands and related volatility indicators
commonly used in technical analysis for stock market data.

Bollinger Bands consist of:
- Middle Band: Simple Moving Average (typically 20 periods)
- Upper Band: Middle Band + (standard deviation multiplier × standard deviation)
- Lower Band: Middle Band - (standard deviation multiplier × standard deviation)

Additional indicators:
- ATR: Average True Range (volatility measure)
- Bandwidth: Bollinger Band width (volatility measure)
- %B: Percent B (position within bands)
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def calculate_bollinger_bands(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data: Price data as pandas Series or DataFrame
        period: Period for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        Tuple containing:
        - upper_band: Upper Bollinger Band
        - middle_band: Middle Band (SMA)
        - lower_band: Lower Bollinger Band

    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(df, period=20, std_dev=2)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    # 计算中轨 (简单移动平均)
    middle_band = series.rolling(window=period, min_periods=1).mean()

    # 计算标准差
    rolling_std = series.rolling(window=period, min_periods=1).std()

    # 计算上轨和下轨
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_bollinger_bandwidth(
    upper_band: pd.Series,
    middle_band: pd.Series,
    lower_band: pd.Series
) -> pd.Series:
    """
    Calculate Bollinger Bandwidth (volatility indicator).

    Args:
        upper_band: Upper Bollinger Band
        middle_band: Middle Bollinger Band
        lower_band: Lower Bollinger Band

    Returns:
        pd.Series: Bandwidth values

    Formula:
        Bandwidth = (Upper Band - Lower Band) / Middle Band

    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(df)
        >>> bandwidth = calculate_bollinger_bandwidth(upper, middle, lower)
    """
    return (upper_band - lower_band) / middle_band


def calculate_bollinger_percent_b(
    data: Union[pd.Series, pd.DataFrame],
    upper_band: pd.Series,
    lower_band: pd.Series,
    column: str = "close"
) -> pd.Series:
    """
    Calculate Bollinger %B (position within bands).

    Args:
        data: Price data as pandas Series or DataFrame
        upper_band: Upper Bollinger Band
        lower_band: Lower Bollinger Band
        column: Column name to use if data is a DataFrame (default: 'close')

    Returns:
        pd.Series: %B values
        - %B > 1: Price above upper band
        - %B = 1: Price at upper band
        - %B = 0.5: Price at middle band
        - %B = 0: Price at lower band
        - %B < 0: Price below lower band

    Formula:
        %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(df)
        >>> percent_b = calculate_bollinger_percent_b(df, upper, lower)
    """
    # 如果传入的是 DataFrame，提取指定列
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    return (series - lower_band) / (upper_band - lower_band)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)

    Returns:
        pd.Series: ATR values

    Example:
        >>> atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
    """
    # 计算前一日收盘价
    prev_close = close.shift(1)

    # 计算 True Range 的三个组成部分
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    # True Range = 三者中的最大值
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算 ATR (使用 EMA 平滑)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channel.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        ema_period: Period for EMA (default: 20)
        atr_period: Period for ATR (default: 10)
        atr_multiplier: ATR multiplier (default: 2.0)

    Returns:
        Tuple containing:
        - upper_channel: Upper Keltner Channel
        - middle_channel: Middle Channel (EMA)
        - lower_channel: Lower Keltner Channel

    Example:
        >>> upper, middle, lower = calculate_keltner_channel(
        ...     df['high'], df['low'], df['close']
        ... )
    """
    # 计算中轨 (EMA)
    middle_channel = close.ewm(span=ema_period, adjust=False).mean()

    # 计算 ATR
    atr = calculate_atr(high, low, close, period=atr_period)

    # 计算上轨和下轨
    upper_channel = middle_channel + (atr * atr_multiplier)
    lower_channel = middle_channel - (atr * atr_multiplier)

    return upper_channel, middle_channel, lower_channel
