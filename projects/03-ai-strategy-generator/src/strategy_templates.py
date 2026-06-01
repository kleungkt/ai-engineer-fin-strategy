"""
Strategy templates module for AI Strategy Generator.

Pre-built Backtrader strategy templates that can be rendered with custom parameters.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "ma_crossover": '''\
import backtrader as bt


class MACrossover(bt.Strategy):
    """Moving Average Crossover strategy."""

    params = (
        ("fast_period", {fast_period}),
        ("slow_period", {slow_period}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()
''',

    "rsi_extreme": '''\
import backtrader as bt


class RSIExtreme(bt.Strategy):
    """RSI mean-reversion strategy: buy oversold, sell overbought."""

    params = (
        ("rsi_period", {rsi_period}),
        ("oversold", {oversold}),
        ("overbought", {overbought}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if not self.position:
            if self.rsi < self.p.oversold:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.rsi > self.p.overbought:
                self.close()
''',

    "macd_signal": '''\
import backtrader as bt


class MACDSignal(bt.Strategy):
    """MACD signal line crossover strategy."""

    params = (
        ("fast_period", {fast_period}),
        ("slow_period", {slow_period}),
        ("signal_period", {signal_period}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            period_signal=self.p.signal_period,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()
''',

    "bollinger_bounce": '''\
import backtrader as bt


class BollingerBounce(bt.Strategy):
    """Bollinger Bands mean-reversion strategy."""

    params = (
        ("period", {period}),
        ("devfactor", {devfactor}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(
            self.data.close, period=self.p.period, devfactor=self.p.devfactor
        )

    def next(self):
        if not self.position:
            if self.data.close < self.boll.lines.bot:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.data.close > self.boll.lines.top:
                self.close()
''',

    "momentum_breakout": '''\
import backtrader as bt


class MomentumBreakout(bt.Strategy):
    """Momentum breakout: buy when price breaks above N-day high."""

    params = (
        ("lookback", {lookback}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.lookback)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.lookback)

    def next(self):
        if not self.position:
            if self.data.close > self.highest[-1]:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.data.close < self.lowest[-1]:
                self.close()
''',

    "dual_thrust": '''\
import backtrader as bt


class DualThrust(bt.Strategy):
    """Dual Thrust breakout strategy by Michael Chalek."""

    params = (
        ("lookback", {lookback}),
        ("k1", {k1}),
        ("k2", {k2}),
        ("stake_pct", {stake_pct}),
    )

    def __init__(self):
        self.range_len = self.p.lookback

    def next(self):
        if len(self) < self.range_len + 1:
            return

        # Calculate range = max(HH-LC, HC-LL) over lookback period
        hh = max(self.data.high.get(size=self.range_len))
        lc = min(self.data.close.get(size=self.range_len))
        hc = max(self.data.close.get(size=self.range_len))
        ll = min(self.data.low.get(size=self.range_len))

        range_val = max(hh - lc, hc - ll)

        open_price = self.data.open[0]
        buy_trigger = open_price + self.p.k1 * range_val
        sell_trigger = open_price - self.p.k2 * range_val

        if not self.position:
            if self.data.close > buy_trigger:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.data.close < sell_trigger:
                self.close()
''',
}

# Default parameters for each template
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "ma_crossover": {"fast_period": 10, "slow_period": 30, "stake_pct": 0.95},
    "rsi_extreme": {"rsi_period": 14, "oversold": 30, "overbought": 70, "stake_pct": 0.95},
    "macd_signal": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "stake_pct": 0.95},
    "bollinger_bounce": {"period": 20, "devfactor": 2.0, "stake_pct": 0.95},
    "momentum_breakout": {"lookback": 20, "stake_pct": 0.95},
    "dual_thrust": {"lookback": 4, "k1": 0.5, "k2": 0.5, "stake_pct": 0.95},
}

# Human-readable descriptions
DESCRIPTIONS: dict[str, str] = {
    "ma_crossover": "Moving Average Crossover - Buy when fast MA crosses above slow MA, sell on cross below.",
    "rsi_extreme": "RSI Extreme - Buy when RSI is oversold, sell when overbought (mean reversion).",
    "macd_signal": "MACD Signal - Buy on MACD line crossing above signal line, sell on cross below.",
    "bollinger_bounce": "Bollinger Bounce - Buy when price touches lower band, sell at upper band.",
    "momentum_breakout": "Momentum Breakout - Buy when price breaks above N-day high, sell on break below N-day low.",
    "dual_thrust": "Dual Thrust - Classic breakout strategy using range-based entry/exit thresholds.",
}


def list_templates() -> list[dict[str, str]]:
    """
    List all available strategy templates.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    return [
        {"name": name, "description": DESCRIPTIONS.get(name, "")}
        for name in sorted(TEMPLATES.keys())
    ]


def get_template(name: str) -> str:
    """
    Get the raw template string for a given strategy name.

    Args:
        name: Template name (e.g., 'ma_crossover')

    Returns:
        Raw template string with {placeholders}

    Raises:
        KeyError: If template name not found
    """
    if name not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return TEMPLATES[name]


def render_template(name: str, params: dict[str, Any] | None = None) -> str:
    """
    Render a strategy template with the given parameters.

    Merges default parameters with user-provided overrides.

    Args:
        name: Template name
        params: Parameter overrides (optional)

    Returns:
        Fully rendered Python code string

    Raises:
        KeyError: If template name not found
    """
    template = get_template(name)
    merged = dict(DEFAULT_PARAMS.get(name, {}))
    if params:
        merged.update(params)
    return template.format(**merged)
