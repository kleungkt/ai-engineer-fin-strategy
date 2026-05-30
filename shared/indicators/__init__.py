"""
Technical Indicators Module
============================

This module provides common technical analysis indicators for stock market data.

Supported indicators:
- Moving Averages: SMA (Simple Moving Average), EMA (Exponential Moving Average)
- Oscillators: MACD (Moving Average Convergence Divergence), RSI (Relative Strength Index)
- Volatility: Bollinger Bands

Usage:
    from shared.indicators import calculate_sma, calculate_ema, calculate_macd, calculate_rsi
    from shared.indicators.bollinger import calculate_bollinger_bands
"""

from .moving_averages import calculate_sma, calculate_ema, calculate_wma
from .oscillators import calculate_macd, calculate_rsi, calculate_stochastic
from .bollinger import calculate_bollinger_bands, calculate_atr

__all__ = [
    # Moving Averages
    "calculate_sma",
    "calculate_ema",
    "calculate_wma",
    # Oscillators
    "calculate_macd",
    "calculate_rsi",
    "calculate_stochastic",
    # Bollinger
    "calculate_bollinger_bands",
    "calculate_atr",
]
