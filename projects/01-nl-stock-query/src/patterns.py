"""
K-line (candlestick) pattern recognition.

Pure-pandas implementation of common candlestick pattern detectors.
No TA-Lib dependency required.

All functions accept a DataFrame with columns: date, open, high, low, close, volume
and return a pd.Series of booleans (True where the pattern is detected).
"""

from __future__ import annotations

import pandas as pd


def _body(row: pd.Series) -> float:
    """Absolute body size (|close - open|)."""
    return abs(row["close"] - row["open"])


def _range(row: pd.Series) -> float:
    """Full candle range (high - low)."""
    return row["high"] - row["low"]


def _is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def _upper_shadow(row: pd.Series) -> float:
    return row["high"] - max(row["close"], row["open"])


def _lower_shadow(row: pd.Series) -> float:
    return min(row["close"], row["open"]) - row["low"]


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect bullish and bearish engulfing patterns.

    A bullish engulfing occurs when a bullish candle's body completely
    engulfs the previous bearish candle's body. A bearish engulfing
    is the opposite.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where an engulfing pattern is detected.
    """
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    # Bullish engulfing: prev bearish, current bullish, current body engulfs prev
    bullish = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    )

    # Bearish engulfing: prev bullish, current bearish, current body engulfs prev
    bearish = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    )

    result = bullish | bearish
    result.iloc[0] = False  # first row has no previous
    return result


def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """Detect doji candles.

    A doji has a very small body relative to its total range.
    Body < threshold * (high - low).

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.
        threshold: Maximum body-to-range ratio to qualify as doji (default 0.1).

    Returns:
        pd.Series of booleans, True where a doji is detected.
    """
    if len(df) < 1:
        return pd.Series(False, index=df.index)

    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()

    # Avoid division by zero
    result = (rng > 0) & (body / rng < threshold)
    return result


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect hammer and inverted hammer patterns.

    Hammer: small body at the top, long lower shadow (≥ 2x body),
    little/no upper shadow. Suggests bullish reversal after downtrend.

    Inverted hammer: small body at the bottom, long upper shadow
    (≥ 2x body), little/no lower shadow.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where a hammer/inverted hammer is detected.
    """
    if len(df) < 1:
        return pd.Series(False, index=df.index)

    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["close", "open"]].max(axis=1)
    lower = df[["close", "open"]].min(axis=1) - df["low"]
    rng = df["high"] - df["low"]

    # Hammer: long lower shadow, small upper shadow, small body
    hammer = (
        (rng > 0)
        & (body > 0)
        & (lower >= 2 * body)
        & (upper <= 0.3 * rng)
    )

    # Inverted hammer: long upper shadow, small lower shadow
    inv_hammer = (
        (rng > 0)
        & (body > 0)
        & (upper >= 2 * body)
        & (lower <= 0.3 * rng)
    )

    return hammer | inv_hammer


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Detect morning star pattern (3-candle bullish reversal).

    1. First candle: bearish with large body
    2. Second candle: small body (star), gaps down
    3. Third candle: bullish, closes above midpoint of first candle

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where a morning star is detected.
    """
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)

    # Use shifted columns for 3-candle lookback
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        # First candle bearish with decent body
        if c1["close"] >= c1["open"]:
            continue
        body1 = abs(c1["close"] - c1["open"])
        if body1 == 0:
            continue

        # Second candle small body
        body2 = abs(c2["close"] - c2["open"])
        rng2 = c2["high"] - c2["low"]
        if rng2 == 0 or body2 / rng2 > 0.3:
            continue

        # Gap down from first to second
        if max(c2["open"], c2["close"]) >= min(c1["open"], c1["close"]):
            continue

        # Third candle bullish and closes above midpoint of first
        mid1 = (c1["open"] + c1["close"]) / 2
        if c3["close"] <= c3["open"]:
            continue
        if c3["close"] < mid1:
            continue

        result.iloc[i] = True

    return result


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Detect evening star pattern (3-candle bearish reversal).

    1. First candle: bullish with large body
    2. Second candle: small body (star), gaps up
    3. Third candle: bearish, closes below midpoint of first candle

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where an evening star is detected.
    """
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        # First candle bullish
        if c1["close"] <= c1["open"]:
            continue
        body1 = abs(c1["close"] - c1["open"])
        if body1 == 0:
            continue

        # Second candle small body
        body2 = abs(c2["close"] - c2["open"])
        rng2 = c2["high"] - c2["low"]
        if rng2 == 0 or body2 / rng2 > 0.3:
            continue

        # Gap up from first to second
        if min(c2["open"], c2["close"]) <= max(c1["open"], c1["close"]):
            continue

        # Third candle bearish and closes below midpoint of first
        mid1 = (c1["open"] + c1["close"]) / 2
        if c3["close"] >= c3["open"]:
            continue
        if c3["close"] > mid1:
            continue

        result.iloc[i] = True

    return result


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Detect three white soldiers pattern (bullish).

    Three consecutive bullish candles, each opening within the prior body
    and closing progressively higher.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where three white soldiers is detected.
    """
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        # All three must be bullish
        if not (_is_bullish(c1) and _is_bullish(c2) and _is_bullish(c3)):
            continue

        # Each opens within the previous body
        if not (min(c1["open"], c1["close"]) < c2["open"] < max(c1["open"], c1["close"])):
            continue
        if not (min(c2["open"], c2["close"]) < c3["open"] < max(c2["open"], c2["close"])):
            continue

        # Each closes progressively higher
        if not (c1["close"] < c2["close"] < c3["close"]):
            continue

        result.iloc[i] = True

    return result


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Detect three black crows pattern (bearish).

    Three consecutive bearish candles, each opening within the prior body
    and closing progressively lower.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        pd.Series of booleans, True where three black crows is detected.
    """
    if len(df) < 3:
        return pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        # All three must be bearish
        if not (_is_bearish(c1) and _is_bearish(c2) and _is_bearish(c3)):
            continue

        # Each opens within the previous body
        if not (min(c1["open"], c1["close"]) < c2["open"] < max(c1["open"], c1["close"])):
            continue
        if not (min(c2["open"], c2["close"]) < c3["open"] < max(c2["open"], c2["close"])):
            continue

        # Each closes progressively lower
        if not (c1["close"] > c2["close"] > c3["close"]):
            continue

        result.iloc[i] = True

    return result


def scan_patterns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Run all candlestick pattern detectors on the DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        Dict mapping pattern name to a boolean Series indicating detection.
    """
    return {
        "engulfing": detect_engulfing(df),
        "doji": detect_doji(df),
        "hammer": detect_hammer(df),
        "morning_star": detect_morning_star(df),
        "evening_star": detect_evening_star(df),
        "three_white_soldiers": detect_three_white_soldiers(df),
        "three_black_crows": detect_three_black_crows(df),
    }


def get_pattern_summary(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of detected candlestick patterns.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume.

    Returns:
        Formatted string summarizing all detected patterns and their counts.
    """
    patterns = scan_patterns(df)
    lines: list[str] = ["📊 K-Line Pattern Summary", "=" * 40]

    total_detected = 0
    for name, series in patterns.items():
        count = int(series.sum())
        total_detected += count
        label = name.replace("_", " ").title()
        if count > 0:
            lines.append(f"  🔹 {label}: {count} occurrences")
        else:
            lines.append(f"  ⚪ {label}: none")

    lines.append(f"\n  Total patterns detected: {total_detected}")
    lines.append(f"  Data points analyzed: {len(df)}")

    if total_detected == 0:
        lines.append("\n  ℹ️  No significant candlestick patterns detected in this dataset.")

    return "\n".join(lines)
