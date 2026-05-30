"""
Stock Screener Module

Takes structured QueryIntent from parser.py and applies it to stock data
using indicators.py. Supports multiple conditions with AND logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Any

from parser import QueryIntent, IndicatorCondition
from indicators import (
    calc_ma,
    calc_ema,
    calc_rsi,
    calc_macd,
    calc_bollinger,
    calc_kdj,
    check_crossover,
    check_crossunder,
)


class ScreenResult(BaseModel):
    """Result of screening a single stock against query conditions."""

    symbol: str
    matched: bool
    indicators: dict[str, Any] = {}
    explanation: str = ""


# ---------------------------------------------------------------------------
# Sample data generation
# ---------------------------------------------------------------------------

def generate_sample_data(symbol: str, days: int = 120) -> pd.DataFrame:
    """
    Generate realistic-looking OHLCV sample data using a random walk with drift.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (used as seed offset for reproducibility).
    days : int
        Number of trading days to generate.

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume
    """
    seed = sum(ord(c) for c in symbol) + 42
    rng = np.random.default_rng(seed)

    # Starting price between 10 and 200
    start_price = rng.uniform(10, 200)

    # Daily returns: slight upward drift ~0.03% per day, std ~1.8%
    drift = 0.0003
    volatility = 0.018
    daily_returns = rng.normal(drift, volatility, size=days)

    # Build close prices from cumulative returns
    close = start_price * np.cumprod(1 + daily_returns)

    # Open = previous close ± small gap
    gap = rng.normal(0, 0.003, size=days)
    open_prices = np.empty(days)
    open_prices[0] = start_price
    open_prices[1:] = close[:-1] * (1 + gap[1:])

    # High / Low from intraday range
    intraday_range = rng.uniform(0.005, 0.03, size=days)
    high = np.maximum(open_prices, close) * (1 + intraday_range / 2)
    low = np.minimum(open_prices, close) * (1 - intraday_range / 2)

    # Volume: base ~1M shares with random variation
    base_volume = rng.uniform(500_000, 5_000_000)
    volume = rng.uniform(base_volume * 0.5, base_volume * 1.5, size=days).astype(int)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    return pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_prices, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# Single-condition evaluator
# ---------------------------------------------------------------------------

def _check_single_condition(
    cond: IndicatorCondition,
    df: pd.DataFrame,
    computed: dict[str, Any],
) -> tuple[bool, str]:
    """
    Evaluate one IndicatorCondition against pre-computed data.

    Returns
    -------
    (passed, detail_string)
    """
    name = cond.name.upper()
    comparison = cond.comparison.lower()
    value = cond.value  # may be None
    params = cond.params or {}

    # ---- RSI ----
    if name == "RSI":
        period = params.get("period", 14)
        rsi_series: pd.Series = computed[f"rsi_{period}"]
        rsi_val = float(rsi_series.iloc[-1])
        if comparison in ("crossover", "crossunder"):
            threshold = value if value is not None else 50.0
            crossed = (
                check_crossover(rsi_series, pd.Series(threshold, index=rsi_series.index)).iloc[-1]
                if comparison == "crossover"
                else check_crossunder(rsi_series, pd.Series(threshold, index=rsi_series.index)).iloc[-1]
            )
            return bool(crossed), f"RSI({period})={rsi_val:.2f}, cross {'above' if comparison == 'crossover' else 'below'} {threshold}"
        if value is not None:
            if comparison == "above":
                passed = rsi_val > value
            elif comparison == "below":
                passed = rsi_val < value
            else:
                passed = False
            return passed, f"RSI({period})={rsi_val:.2f} {'>' if comparison == 'above' else '<'} {value}"
        return False, f"RSI({period})={rsi_val:.2f}"

    # ---- MACD ----
    if name == "MACD":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        sig = params.get("signal", 9)
        key = f"macd_{fast}_{slow}_{sig}"
        macd_dict = computed[key]
        macd_line = macd_dict["macd"]
        signal_line = macd_dict["signal"]
        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])

        if comparison == "crossover":
            crossed = bool(check_crossover(macd_line, signal_line).iloc[-1])
            return crossed, f"MACD={macd_val:.4f}, Signal={sig_val:.4f} (crossover {'✓' if crossed else '✗'})"
        elif comparison == "crossunder":
            crossed = bool(check_crossunder(macd_line, signal_line).iloc[-1])
            return crossed, f"MACD={macd_val:.4f}, Signal={sig_val:.4f} (crossunder {'✓' if crossed else '✗'})"
        elif comparison == "above":
            return macd_val > sig_val, f"MACD={macd_val:.4f} {'>' if macd_val > sig_val else '<='} Signal={sig_val:.4f}"
        elif comparison == "below":
            return macd_val < sig_val, f"MACD={macd_val:.4f} {'<' if macd_val < sig_val else '>='} Signal={sig_val:.4f}"

    # ---- SMA / MA ----
    if name in ("SMA", "MA"):
        period = params.get("period", 20)
        ma_series = computed[f"ma_{period}"]
        ma_val = float(ma_series.iloc[-1])
        close_val = float(df["close"].iloc[-1])
        if comparison == "above":
            # "close above MA"
            passed = close_val > ma_val
            return passed, f"Close={close_val:.2f} {'>' if passed else '<='} SMA({period})={ma_val:.2f}"
        elif comparison == "below":
            passed = close_val < ma_val
            return passed, f"Close={close_val:.2f} {'<' if passed else '>='} SMA({period})={ma_val:.2f}"
        elif comparison == "crossover":
            crossed = bool(check_crossover(df["close"], ma_series).iloc[-1])
            return crossed, f"Close crossed above SMA({period})={ma_val:.2f} ({'✓' if crossed else '✗'})"
        elif comparison == "crossunder":
            crossed = bool(check_crossunder(df["close"], ma_series).iloc[-1])
            return crossed, f"Close crossed below SMA({period})={ma_val:.2f} ({'✓' if crossed else '✗'})"

    # ---- EMA ----
    if name == "EMA":
        period = params.get("period", 20)
        ema_series = computed[f"ema_{period}"]
        ema_val = float(ema_series.iloc[-1])
        close_val = float(df["close"].iloc[-1])
        if comparison == "above":
            passed = close_val > ema_val
            return passed, f"Close={close_val:.2f} {'>' if passed else '<='} EMA({period})={ema_val:.2f}"
        elif comparison == "below":
            passed = close_val < ema_val
            return passed, f"Close={close_val:.2f} {'<' if passed else '>='} EMA({period})={ema_val:.2f}"
        elif comparison == "crossover":
            crossed = bool(check_crossover(df["close"], ema_series).iloc[-1])
            return crossed, f"Close crossed above EMA({period})={ema_val:.2f} ({'✓' if crossed else '✗'})"
        elif comparison == "crossunder":
            crossed = bool(check_crossunder(df["close"], ema_series).iloc[-1])
            return crossed, f"Close crossed below EMA({period})={ema_val:.2f} ({'✓' if crossed else '✗'})"

    # ---- Bollinger Bands ----
    if name in ("BOLL", "BOLLINGER"):
        period = params.get("period", 20)
        std_dev = params.get("std", 2)
        key = f"boll_{period}_{std_dev}"
        bb = computed[key]
        close_val = float(df["close"].iloc[-1])
        upper = float(bb["upper"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        middle = float(bb["middle"].iloc[-1])

        if comparison == "above":
            # Close above upper band → breakout
            passed = close_val > upper
            return passed, f"Close={close_val:.2f} {'>' if passed else '<='} BB_Upper={upper:.2f}"
        elif comparison == "below":
            # Close below lower band → breakdown
            passed = close_val < lower
            return passed, f"Close={close_val:.2f} {'<' if passed else '>='} BB_Lower={lower:.2f}"
        elif comparison == "crossover":
            crossed = bool(check_crossover(df["close"], bb["upper"]).iloc[-1])
            return crossed, f"Close crossed above BB_Upper={upper:.2f} ({'✓' if crossed else '✗'})"
        elif comparison == "crossunder":
            crossed = bool(check_crossunder(df["close"], bb["lower"]).iloc[-1])
            return crossed, f"Close crossed below BB_Lower={lower:.2f} ({'✓' if crossed else '✗'})"

    # ---- KDJ ----
    if name in ("KDJ", "KD"):
        n = params.get("n", params.get("period", 9))
        m1 = params.get("m1", 3)
        m2 = params.get("m2", 3)
        key = f"kdj_{n}_{m1}_{m2}"
        kdj = computed[key]
        k_val = float(kdj["k"].iloc[-1])
        d_val = float(kdj["d"].iloc[-1])

        if comparison == "crossover":
            crossed = bool(check_crossover(kdj["k"], kdj["d"]).iloc[-1])
            return crossed, f"KDJ K={k_val:.2f}, D={d_val:.2f} (crossover {'✓' if crossed else '✗'})"
        elif comparison == "crossunder":
            crossed = bool(check_crossunder(kdj["k"], kdj["d"]).iloc[-1])
            return crossed, f"KDJ K={k_val:.2f}, D={d_val:.2f} (crossunder {'✓' if crossed else '✗'})"
        elif comparison == "above" and value is not None:
            passed = k_val > value
            return passed, f"KDJ K={k_val:.2f} {'>' if passed else '<='} {value}"
        elif comparison == "below" and value is not None:
            passed = k_val < value
            return passed, f"KDJ K={k_val:.2f} {'<' if passed else '>='} {value}"

    # ---- Price (raw close) ----
    if name == "PRICE":
        close_val = float(df["close"].iloc[-1])
        if value is not None:
            if comparison == "above":
                passed = close_val > value
            elif comparison == "below":
                passed = close_val < value
            else:
                passed = False
            return passed, f"Price={close_val:.2f} {'>' if comparison == 'above' else '<'} {value}"

    return False, f"Unsupported condition: {name} {comparison}"


# ---------------------------------------------------------------------------
# Pre-compute indicators needed by all conditions across all stocks
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame, conditions: list[IndicatorCondition]) -> dict[str, Any]:
    """Compute every indicator referenced by *conditions* on a single DataFrame."""
    computed: dict[str, Any] = {}

    for cond in conditions:
        name = cond.name.upper()
        params = cond.params or {}

        if name == "RSI":
            period = params.get("period", 14)
            key = f"rsi_{period}"
            if key not in computed:
                computed[key] = calc_rsi(df, period=period)

        elif name == "MACD":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            sig = params.get("signal", 9)
            key = f"macd_{fast}_{slow}_{sig}"
            if key not in computed:
                computed[key] = calc_macd(df, fast=fast, slow=slow, signal=sig)

        elif name in ("SMA", "MA"):
            period = params.get("period", 20)
            key = f"ma_{period}"
            if key not in computed:
                computed[key] = calc_ma(df, period=period)

        elif name == "EMA":
            period = params.get("period", 20)
            key = f"ema_{period}"
            if key not in computed:
                computed[key] = calc_ema(df, period=period)

        elif name in ("BOLL", "BOLLINGER"):
            period = params.get("period", 20)
            std_dev = params.get("std", 2)
            key = f"boll_{period}_{std_dev}"
            if key not in computed:
                computed[key] = calc_bollinger(df, period=period, std=std_dev)

        elif name in ("KDJ", "KD"):
            n = params.get("n", params.get("period", 9))
            m1 = params.get("m1", 3)
            m2 = params.get("m2", 3)
            key = f"kdj_{n}_{m1}_{m2}"
            if key not in computed:
                computed[key] = calc_kdj(df, n=n, m1=m1, m2=m2)

    return computed


# ---------------------------------------------------------------------------
# Snapshot latest indicator values for output
# ---------------------------------------------------------------------------

def _build_indicator_snapshot(computed: dict[str, Any]) -> dict[str, Any]:
    """Extract the latest value from every computed indicator for display."""
    snapshot: dict[str, Any] = {}
    for key, val in computed.items():
        if isinstance(val, pd.Series):
            raw = val.iloc[-1]
            snapshot[key] = round(float(raw), 4) if pd.notna(raw) else None
        elif isinstance(val, dict):
            snapshot[key] = {
                k: round(float(v.iloc[-1]), 4) if pd.notna(v.iloc[-1]) else None
                for k, v in val.items()
                if isinstance(v, pd.Series)
            }
    return snapshot


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------

def screen_stocks(
    query: QueryIntent,
    stock_data: dict[str, pd.DataFrame],
) -> list[ScreenResult]:
    """
    Screen stocks against a parsed query intent.

    Parameters
    ----------
    query : QueryIntent
        Parsed user query with indicator conditions.
    stock_data : dict[str, pd.DataFrame]
        Mapping of symbol -> OHLCV DataFrame.

    Returns
    -------
    list[ScreenResult]
        One result per stock. Stocks where ALL conditions pass have matched=True.
    """
    results: list[ScreenResult] = []
    conditions = query.indicators  # list[IndicatorCondition] from parser

    if not conditions:
        for symbol in stock_data:
            results.append(
                ScreenResult(symbol=symbol, matched=True, explanation="No conditions specified — all stocks returned.")
            )
        return results

    for symbol, df in stock_data.items():
        if len(df) < 5:
            results.append(
                ScreenResult(symbol=symbol, matched=False, explanation="Insufficient data.")
            )
            continue

        # Pre-compute all needed indicators once per stock
        computed = _compute_indicators(df, conditions)

        condition_details: list[str] = []
        all_passed = True

        for cond in conditions:
            passed, detail = _check_single_condition(cond, df, computed)
            condition_details.append(
                f"{cond.name} {cond.comparison} {cond.value if cond.value is not None else ''}: "
                f"{detail} → {'✓' if passed else '✗'}"
            )
            if not passed:
                all_passed = False

        snapshot = _build_indicator_snapshot(computed)
        explanation = "; ".join(condition_details)

        results.append(
            ScreenResult(
                symbol=symbol,
                matched=all_passed,
                indicators=snapshot,
                explanation=explanation,
            )
        )

    return results
