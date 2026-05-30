"""
Simple backtesting engine wrapping Backtrader.
Provides strategy factory, runner, and result extraction.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Optional

import backtrader as bt
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class TradeRecord(BaseModel):
    entry_date: str
    exit_date: Optional[str] = None
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    reason: str = ""


class BacktestResult(BaseModel):
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    total_trades: int = 0
    trade_log: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared base strategy with trade logging helpers
# ---------------------------------------------------------------------------

class _BaseStrategy(bt.Strategy):
    """Base class that provides trade logging utilities."""

    trade_log: list[dict[str, Any]]
    _entry_info: dict[str, Any] | None

    def __init__(self) -> None:
        super().__init__()
        self.trade_log = []
        self._entry_info = None

    def _enter(self, reason: str) -> None:
        dt_val = self.datas[0].datetime.date(0)
        price = self.datas[0].close[0]
        self._entry_info = {"entry_date": str(dt_val), "entry_price": price, "reason": reason}
        self.buy()

    def _exit(self, reason: str) -> None:
        dt_val = self.datas[0].datetime.date(0)
        price = self.datas[0].close[0]
        if self._entry_info:
            entry = self._entry_info
            pnl = price - entry["entry_price"]
            record = {
                "entry_date": entry["entry_date"],
                "exit_date": str(dt_val),
                "entry_price": entry["entry_price"],
                "exit_price": price,
                "pnl": round(pnl, 4),
                "reason": f"Entry: {entry['reason']}; Exit: {reason}",
            }
            self.trade_log.append(record)
            self._entry_info = None
        self.sell()


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class MACrossover(_BaseStrategy):
    """Dual moving-average crossover strategy."""

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self) -> None:
        if self.crossover > 0 and not self.position:
            self._enter(f"MA fast({self.fast_ma.p.period}) crossed above MA slow({self.slow_ma.p.period})")
        elif self.crossover < 0 and self.position:
            self._exit(f"MA fast crossed below MA slow")


class RSIExtreme(_BaseStrategy):
    """RSI overbought / oversold strategy."""

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> None:
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=period)
        self.oversold = oversold
        self.overbought = overbought

    def next(self) -> None:
        if self.rsi < self.oversold and not self.position:
            self._enter(f"RSI({self.rsi.p.period}) = {self.rsi[0]:.1f} < {self.oversold} (oversold)")
        elif self.rsi > self.overbought and self.position:
            self._exit(f"RSI = {self.rsi[0]:.1f} > {self.overbought} (overbought)")


class MACDSignal(_BaseStrategy):
    """MACD line crossing signal line strategy."""

    def __init__(self) -> None:
        super().__init__()
        self.macd = bt.indicators.MACD(self.data.close)
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self) -> None:
        if self.crossover > 0 and not self.position:
            self._enter("MACD crossed above signal line")
        elif self.crossover < 0 and self.position:
            self._exit("MACD crossed below signal line")


class BollingerBounce(_BaseStrategy):
    """Buy when price touches lower Bollinger Band, sell at upper band."""

    def __init__(self, period: int = 20, devfactor: float = 2.0) -> None:
        super().__init__()
        self.boll = bt.indicators.BollingerBands(
            self.data.close, period=period, devfactor=devfactor
        )

    def next(self) -> None:
        if self.data.close[0] < self.boll.lines.bot[0] and not self.position:
            self._enter("Price below lower Bollinger Band")
        elif self.data.close[0] > self.boll.lines.top[0] and self.position:
            self._exit("Price above upper Bollinger Band")


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, type[_BaseStrategy]] = {
    "ma_crossover": MACrossover,
    "rsi_extreme": RSIExtreme,
    "macd_signal": MACDSignal,
    "bollinger_bounce": BollingerBounce,
}


def build_strategy(strategy_type: str, params: dict | None = None) -> type[bt.Strategy]:
    """Return a Backtrader Strategy *class* for the given type and params.

    Args:
        strategy_type: One of 'ma_crossover', 'rsi_extreme', 'macd_signal', 'bollinger_bounce'.
        params: Keyword arguments forwarded to the strategy constructor.

    Returns:
        A Strategy subclass ready to be added to Cerebro.
    """
    if strategy_type not in _STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{strategy_type}'. Choose from {list(_STRATEGY_MAP)}")
    cls = _STRATEGY_MAP[strategy_type]

    # We return a partial-application wrapper so params are baked in
    params = params or {}

    class _Configured(cls):
        def __init__(self) -> None:
            super().__init__(**params)

    _Configured.__name__ = f"Configured{cls.__name__}"
    return _Configured


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

def run_backtest(
    data: pd.DataFrame,
    strategy_type: str,
    params: dict | None = None,
    initial_cash: float = 100_000,
) -> BacktestResult:
    """Run a backtest on *data* using the specified strategy.

    Args:
        data: DataFrame with columns date, open, high, low, close, volume.
        strategy_type: Strategy identifier.
        params: Strategy parameters.
        initial_cash: Starting cash for the broker.

    Returns:
        BacktestResult with performance metrics and trade log.
    """
    params = params or {}

    # Backtrader needs at least ~50 rows for indicators to work
    if len(data) < 50:
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            total_trades=0,
            trade_log=[],
        )

    # Prepare data
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1 %

    # Data feed
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # Strategy
    strategy_cls = build_strategy(strategy_type, params)
    cerebro.addstrategy(strategy_cls)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run
    results = cerebro.run()
    strat = results[0]

    # --- Extract metrics ---
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    ret_analysis = strat.analyzers.returns.get_analysis()
    trade_analysis = strat.analyzers.trades.get_analysis()

    sharpe_val = sharpe_analysis.get("sharperatio")
    max_dd = dd_analysis.get("max", {}).get("drawdown", 0.0)
    max_dd_len = dd_analysis.get("max", {}).get("len", 0)
    total_return = ret_analysis.get("rtot", 0.0) * 100  # percent
    # Annualized rough approximation
    n_bars = len(df)
    annual_return = ((1 + ret_analysis.get("rtot", 0.0)) ** (252 / max(n_bars, 1)) - 1) * 100

    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades else 0.0

    trade_log: list[dict[str, Any]] = getattr(strat, "trade_log", [])

    return BacktestResult(
        total_return=round(total_return, 2),
        annual_return=round(annual_return, 2),
        sharpe_ratio=round(sharpe_val, 4) if sharpe_val is not None else None,
        max_drawdown=round(max_dd, 2),
        max_drawdown_duration=max_dd_len,
        win_rate=round(win_rate, 2),
        total_trades=total_trades,
        trade_log=trade_log,
    )


# ---------------------------------------------------------------------------
# NL query → strategy mapping
# ---------------------------------------------------------------------------

def map_query_to_strategy(query: Any) -> tuple[str, dict]:
    """Map a parsed NL query intent to a strategy type and parameters.

    Expects *query* to have attributes ``intent_type``, ``indicators``,
    ``time_range``, etc.  (as produced by the NL parser).

    Args:
        query: Parsed query intent object.

    Returns:
        Tuple of (strategy_type, params_dict).
    """
    indicators: list[str] = getattr(query, "indicators", []) or []
    intent_type: str = getattr(query, "intent_type", "") or ""
    ind_lower = [i.lower() for i in indicators]

    # MACD
    if any("macd" in i for i in ind_lower):
        return ("macd_signal", {})

    # RSI
    if any("rsi" in i for i in ind_lower):
        params: dict[str, Any] = {}
        # Try to detect custom thresholds from intent_type text
        text = intent_type.lower()
        for token in text.split():
            if token.isdigit():
                val = int(token)
                if val < 50:
                    params.setdefault("oversold", val)
                else:
                    params.setdefault("overbought", val)
        return ("rsi_extreme", params)

    # Bollinger
    if any("bollinger" in i or "boll" in i for i in ind_lower):
        return ("bollinger_bounce", {})

    # Moving average / golden cross / death cross
    if any("ma" in i or "moving average" in i or "sma" in i or "ema" in i for i in ind_lower):
        return ("ma_crossover", {})

    # Fallback: default to MA crossover
    return ("ma_crossover", {})
