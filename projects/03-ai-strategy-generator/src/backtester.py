"""
Backtester module for AI Strategy Generator.

Runs backtests on generated Backtrader strategy code against historical data.
"""

import datetime
import sys
import io
import contextlib
from typing import Any, Optional

import backtrader as bt
import pandas as pd
from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    """Result of a backtest run."""

    total_return: float = Field(description="Total return as a decimal (e.g., 0.15 for 15%)")
    annual_return: float = Field(description="Annualized return as a decimal")
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown as a decimal")
    win_rate: float = Field(description="Win rate as a decimal (0-1)")
    total_trades: int = Field(description="Total number of closed trades")
    trade_log: list[dict[str, Any]] = Field(default_factory=list, description="Log of executed trades")


class PandasDataFeed(bt.feeds.PandasData):
    """Backtrader data feed from a pandas DataFrame with date, open, high, low, close, volume."""

    params = (
        ("datetime", "date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class TradeAnalyzer(bt.Analyzer):
    """Custom analyzer to collect trade statistics."""

    def __init__(self):
        self.trades: list[dict[str, Any]] = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "pnl": round(trade.pnl, 2),
                "pnlcomm": round(trade.pnlcomm, 2),
                "bar_open": trade.baropen,
                "bar_close": trade.barclose,
                "bar_len": trade.barlen,
            })

    def get_analysis(self):
        return {"trades": self.trades}


def run_backtest(
    data: pd.DataFrame,
    strategy_code: str,
    params: Optional[dict] = None,
    initial_cash: float = 100000,
) -> BacktestResult:
    """
    Run a backtest using dynamically loaded strategy code.

    Args:
        data: DataFrame with columns: date, open, high, low, close, volume
        strategy_code: Python code string defining a Backtrader Strategy class
        params: Optional parameters to pass to the strategy
        initial_cash: Starting cash for the backtest

    Returns:
        BacktestResult with performance metrics
    """
    # Suppress backtrader output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        # Create a namespace and exec the strategy code into it
        namespace: dict[str, Any] = {}
        exec(strategy_code, namespace)  # noqa: S102

        # Find the strategy class
        strategy_cls = None
        for obj in namespace.values():
            if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy:
                strategy_cls = obj
                break

        if strategy_cls is None:
            return BacktestResult(
                total_return=0.0,
                annual_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                trade_log=[],
            )

        # Set up Cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)

        # Add data feed
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        data_feed = bt.feeds.PandasData(
            dataname=df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
        cerebro.adddata(data_feed)

        # Add strategy with optional params
        if params:
            cerebro.addstrategy(strategy_cls, **params)
        else:
            cerebro.addstrategy(strategy_cls)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(TradeAnalyzer, _name="trade_log")

        # Run
        results = cerebro.run()
        strat = results[0]

        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash

        # Annual return estimate
        n_bars = len(df)
        if n_bars > 1:
            years = n_bars / 252
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        else:
            annual_return = 0.0

        # Sharpe ratio
        sharpe_data = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_data.get("sharperatio", 0.0) or 0.0

        # Max drawdown
        dd_data = strat.analyzers.drawdown.get_analysis()
        max_drawdown = dd_data.get("max", {}).get("drawdown", 0.0) or 0.0
        max_drawdown = max_drawdown / 100.0

        # Trade stats
        trade_data = strat.analyzers.trades.get_analysis()
        total_closed = trade_data.get("total", {}).get("closed", 0)
        won = trade_data.get("won", {}).get("total", 0)
        win_rate = (won / total_closed) if total_closed > 0 else 0.0

        # Trade log
        trade_log_data = strat.analyzers.trade_log.get_analysis()
        trade_log = trade_log_data.get("trades", [])

        return BacktestResult(
            total_return=round(total_return, 6),
            annual_return=round(annual_return, 6),
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown=round(max_drawdown, 6),
            win_rate=round(win_rate, 4),
            total_trades=total_closed,
            trade_log=trade_log,
        )

    except Exception as e:
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            trade_log=[{"error": str(e)}],
        )
    finally:
        sys.stdout = old_stdout
