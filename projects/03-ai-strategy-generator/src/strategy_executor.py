"""
Strategy Executor Module
========================
Safely executes generated strategy code and runs backtesting.
Bridges code generation and backtesting with sandboxed execution.
"""

from __future__ import annotations

import math
import types
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. ExecutionResult – Pydantic model for backtest output
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Container for strategy execution and backtest results."""

    success: bool = False
    error_message: Optional[str] = None
    signals: Optional[Any] = Field(default=None, description="DataFrame with signal column, or None")
    trades: list[dict] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    equity_curve: list[float] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# 2. StrategySandbox – Restricted code execution
# ---------------------------------------------------------------------------

# Builtins that are explicitly blocked for safety
_BLOCKED_BUILTINS: set[str] = {
    "open", "exec", "eval", "compile", "__import__", "breakpoint",
    "exit", "quit", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "classmethod", "staticmethod",
    "property", "super", "type", "object",
    "memoryview", "bytearray", "bytes",
    "input", "print",
}


class StrategySandbox:
    """Safely execute generated strategy code with restricted builtins."""

    def __init__(self, allowed_modules: Optional[list[str]] = None) -> None:
        self.allowed_modules: list[str] = allowed_modules or [
            "pandas", "numpy", "math",
        ]

    # -- helpers -----------------------------------------------------------

    def _build_safe_builtins(self) -> dict[str, Any]:
        """Return a dict of safe builtins (blocked ones removed)."""
        import builtins as _builtins  # noqa: WPS433

        safe: dict[str, Any] = {}
        for name, obj in vars(_builtins).items():
            if name in _BLOCKED_BUILTINS:
                continue
            safe[name] = obj
        return safe

    def _build_restricted_globals(self) -> dict[str, Any]:
        """Build the global namespace available inside executed code."""
        import importlib

        restricted_globals: dict[str, Any] = {
            "__builtins__": self._build_safe_builtins(),
        }

        # Pre-import allowed modules
        for mod_name in self.allowed_modules:
            try:
                restricted_globals[mod_name] = importlib.import_module(mod_name)
            except ImportError:
                pass  # Module not available – skip silently

        # Convenience aliases
        restricted_globals["pd"] = restricted_globals.get("pandas")
        restricted_globals["np"] = restricted_globals.get("numpy")

        return restricted_globals

    # -- public API --------------------------------------------------------

    def execute_code(self, code: str, class_name: str) -> type:
        """
        Safely load *code* in a restricted environment and return the class
        identified by *class_name*.

        Raises
        ------
        ValueError  – if the class is not found in the executed code.
        """
        restricted_globals = self._build_restricted_globals()
        restricted_locals: dict[str, Any] = {}

        exec(code, restricted_globals, restricted_locals)  # noqa: S102

        strategy_cls: type | None = restricted_locals.get(class_name)
        if strategy_cls is None:
            raise ValueError(
                f"Class '{class_name}' not found in the provided code."
            )
        return strategy_cls

    def run_signals(
        self,
        strategy_cls: type,
        data: pd.DataFrame,
        params: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Instantiate *strategy_cls*, call ``generate_signals(data)``, and
        return the DataFrame with a ``signal`` column added.

        Raises
        ------
        TypeError  – if the output is not a DataFrame or lacks 'signal'.
        """
        instance = strategy_cls(**(params or {}))

        result: pd.DataFrame = instance.generate_signals(data.copy())

        if not isinstance(result, pd.DataFrame):
            raise TypeError("generate_signals() must return a pandas DataFrame.")
        if "signal" not in result.columns:
            raise TypeError("generate_signals() output must contain a 'signal' column.")

        return result


# ---------------------------------------------------------------------------
# 3. simulate_trades – Long-only backtest engine
# ---------------------------------------------------------------------------

def simulate_trades(
    data: pd.DataFrame,
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
) -> tuple[list[dict], list[float]]:
    """
    Simulate a simple long-only strategy on *data* which must contain
    ``'close'`` and ``'signal'`` columns.

    Parameters
    ----------
    data : DataFrame with at least ``close`` and ``signal`` columns.
    initial_cash : Starting cash balance.
    commission : Fractional commission per trade (e.g. 0.001 = 0.1 %).

    Returns
    -------
    trades : list of trade dicts (entry/exit dates, prices, pnl, return %)
    equity_curve : list of daily portfolio values
    """
    if "close" not in data.columns:
        raise ValueError("Data must contain a 'close' column.")
    if "signal" not in data.columns:
        raise ValueError("Data must contain a 'signal' column.")

    cash: float = initial_cash
    position: float = 0.0  # number of shares held
    entry_price: float = 0.0
    entry_date: Any = None

    trades: list[dict] = []
    equity_curve: list[float] = []

    # Determine which column holds dates – prefer explicit date cols, fall
    # back to index.
    has_date_col = "date" in data.columns

    for idx in range(len(data)):
        row = data.iloc[idx]
        price: float = float(row["close"])
        signal: float = float(row["signal"])
        date_val = row["date"] if has_date_col else data.index[idx]

        # ---- BUY signal when flat --------------------------------------
        if signal == 1 and position == 0:
            cost = cash * commission
            shares = (cash - cost) / price
            if shares > 0:
                position = shares
                entry_price = price
                entry_date = date_val
                cash = 0.0

        # ---- SELL signal when long -------------------------------------
        elif signal == -1 and position > 0:
            proceeds = position * price
            cost = proceeds * commission
            cash = proceeds - cost

            pnl = cash - initial_cash  # simplified – will be overwritten per trade
            trade_cash_before = position * entry_price
            pnl = (price - entry_price) * position - (trade_cash_before * commission + proceeds * commission)
            ret_pct = (price - entry_price) / entry_price - 2 * commission

            trades.append(
                {
                    "entry_date": str(entry_date),
                    "exit_date": str(date_val),
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(price, 6),
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct * 100, 4),
                }
            )
            position = 0.0
            entry_price = 0.0

        # ---- Daily portfolio value --------------------------------------
        portfolio_value: float = cash + position * price
        equity_curve.append(round(portfolio_value, 2))

    return trades, equity_curve


# ---------------------------------------------------------------------------
# 4. calculate_metrics – Performance analytics
# ---------------------------------------------------------------------------

def calculate_metrics(
    trades: list[dict],
    equity_curve: list[float],
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    """
    Compute standard backtest metrics from trade log and equity curve.

    Returns a dict with keys:
        total_return, annual_return, sharpe_ratio, max_drawdown,
        max_drawdown_duration, win_rate, total_trades,
        avg_trade_return, profit_factor
    """
    metrics: dict[str, Any] = {
        "total_trades": len(trades),
    }

    # ---- Equity-curve based metrics ------------------------------------
    if not equity_curve:
        metrics.update(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            profit_factor=0.0,
        )
        return metrics

    eq = pd.Series(equity_curve, dtype=float)
    initial_value = eq.iloc[0]
    final_value = eq.iloc[-1]

    # Total return
    total_return = (final_value - initial_value) / initial_value if initial_value else 0.0
    metrics["total_return"] = round(total_return * 100, 4)

    # Daily returns for Sharpe / drawdown
    daily_returns = eq.pct_change().dropna()

    # Annualised return (assume ~252 trading days)
    n_days = len(eq)
    years = n_days / 252 if n_days > 0 else 1.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    metrics["annual_return"] = round(annual_return * 100, 4)

    # Sharpe ratio
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess = daily_returns - daily_rf
        sharpe = (excess.mean() / excess.std()) * math.sqrt(252)
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = round(sharpe, 4)

    # Max drawdown & duration
    cum_max = eq.cummax()
    drawdown = (eq - cum_max) / cum_max
    max_dd = float(drawdown.min())
    metrics["max_drawdown"] = round(abs(max_dd) * 100, 4)

    # Drawdown duration (longest streak below previous peak)
    in_dd = eq < cum_max
    dd_groups = (~in_dd).cumsum()
    if in_dd.any():
        dd_durations = in_dd.groupby(dd_groups).sum()
        metrics["max_drawdown_duration"] = int(dd_durations.max())
    else:
        metrics["max_drawdown_duration"] = 0

    # ---- Trade-level metrics -------------------------------------------
    if trades:
        returns = [t["return_pct"] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        metrics["win_rate"] = round(len(wins) / len(trades) * 100, 2) if trades else 0.0
        metrics["avg_trade_return"] = round(sum(returns) / len(returns), 4) if returns else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        metrics["profit_factor"] = (
            round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        )
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_trade_return"] = 0.0
        metrics["profit_factor"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# 5. run_strategy – High-level orchestrator
# ---------------------------------------------------------------------------

def run_strategy(
    code: str,
    data: pd.DataFrame,
    initial_cash: float = 100_000.0,
) -> ExecutionResult:
    """
    End-to-end strategy runner:

    1. Load and execute *code* in a sandbox.
    2. Generate signals on *data*.
    3. Simulate trades.
    4. Compute performance metrics.

    Returns an :class:`ExecutionResult` with ``success=True`` on success,
    or ``success=False`` and an ``error_message`` on failure.
    """
    try:
        # 1. Discover class name from code (first class definition)
        import re

        class_match = re.search(r"class\s+(\w+)\s*[\(:]", code)
        if not class_match:
            return ExecutionResult(
                success=False,
                error_message="No class definition found in the provided code.",
            )
        class_name = class_match.group(1)

        # 2. Sandbox execution
        sandbox = StrategySandbox()
        strategy_cls = sandbox.execute_code(code, class_name)

        # 3. Generate signals
        signals_df = sandbox.run_signals(strategy_cls, data)

        # 4. Simulate trades
        trades, equity_curve = simulate_trades(signals_df, initial_cash=initial_cash)

        # 5. Compute metrics
        metrics = calculate_metrics(trades, equity_curve)

        return ExecutionResult(
            success=True,
            signals=signals_df,
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    except Exception as exc:
        return ExecutionResult(
            success=False,
            error_message=f"{type(exc).__name__}: {exc}",
        )
