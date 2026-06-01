"""
Backtest engine module.
回测引擎模块 - 动态执行策略代码并运行Backtrader回测。
"""

import io
import logging
import contextlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import backtrader as bt

from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class PandasDataFeed(bt.feeds.PandasData):
    """Backtrader数据源，适配标准OHLCV DataFrame。"""

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class TradeCollector(bt.Analyzer):
    """收集交易记录的分析器。"""

    def __init__(self):
        self.trades: list[dict[str, Any]] = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "entry_date": bt.num2date(trade.dtopen).strftime("%Y-%m-%d"),
                "exit_date": bt.num2date(trade.dtclose).strftime("%Y-%m-%d"),
                "pnl": float(trade.pnl),
                "pnlcomm": float(trade.pnlcomm),
                "size": trade.size,
                "price": float(trade.price),
            })

    def get_analysis(self):
        return {"trades": self.trades}


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """准备回测数据，确保格式正确。"""
    df = data.copy()

    # 确保列名为小写
    df.columns = [c.lower() for c in df.columns]

    # 确保有必要的列
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必要的价格列: {col}")

    # 如果没有volume列，添加默认值
    if "volume" not in df.columns:
        df["volume"] = 0

    # 确保索引是DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"], errors="ignore")
        else:
            df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

    return df


def _execute_strategy_code(strategy_code: str, params: dict[str, Any]) -> type:
    """
    动态执行策略代码，返回策略类。

    策略代码应该定义一个名为 GeneratedStrategy 的类，
    继承自 bt.Strategy。
    """
    # 构建执行环境
    exec_globals = {
        "bt": bt,
        "np": np,
        "pd": pd,
        "params": params,
    }

    try:
        exec(strategy_code, exec_globals)
    except Exception as e:
        raise ValueError(f"策略代码执行失败: {e}")

    # 查找策略类
    strategy_class = exec_globals.get("GeneratedStrategy")
    if strategy_class is None:
        # 尝试查找任何继承自 bt.Strategy 的类
        for name, obj in exec_globals.items():
            if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy:
                strategy_class = obj
                break

    if strategy_class is None:
        raise ValueError("策略代码中未找到有效的策略类 (需继承 bt.Strategy)")

    return strategy_class


def _extract_portfolio_values(cerebro: bt.Cerebro) -> pd.Series:
    """提取投资组合价值时间序列。"""
    values = []
    for strat in cerebro.runstrats:
        if hasattr(strat[0], "broker"):
            # 从观察者中获取
            pass

    # 使用cerebro的observers
    try:
        strat = cerebro.runstrats[0][0]
        # 尝试获取analyzers的数据
        returns_analyzer = strat.analyzers.getbyname("returns")
    except Exception:
        pass

    return pd.Series(dtype=float)


def run_strategy_backtest(
    data: pd.DataFrame,
    strategy_code: str,
    params: dict[str, Any],
    initial_cash: float = 100000,
) -> dict[str, Any]:
    """
    Run a backtest with dynamically generated strategy code.

    Executes the provided strategy code, runs it through Backtrader,
    and returns comprehensive results including metrics and trade log.

    Args:
        data: OHLCV price data
        strategy_code: Python code defining a Backtrader strategy class
        params: Strategy parameters to inject
        initial_cash: Initial portfolio cash amount

    Returns:
        Dictionary with comprehensive backtest results
    """
    logger.info(f"开始回测，初始资金: {initial_cash:,.0f}")

    # 准备数据
    df = _prepare_data(data)

    # 动态执行策略代码获取策略类
    strategy_class = _execute_strategy_code(strategy_code, params)

    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()

    # 添加数据源
    data_feed = PandasDataFeed(dataname=df)
    cerebro.adddata(data_feed)

    # 添加策略
    cerebro.addstrategy(strategy_class, **params)

    # 设置初始资金
    cerebro.broker.setcash(initial_cash)

    # 设置佣金 (万分之三，中国市场标准)
    cerebro.broker.setcommission(commission=0.0003)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(TradeCollector, _name="trade_collector")

    # 运行回测
    final_value = cerebro.broker.getvalue()
    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        return {
            "error": str(e),
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    strat = results[0]

    # 提取分析器结果
    returns_analysis = strat.analyzers.returns.get_analysis()
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    trade_analysis = strat.analyzers.trades.get_analysis()
    trade_collector = strat.analyzers.trade_collector.get_analysis()

    # 计算收益
    total_return = (final_value - initial_cash) / initial_cash
    annual_return = returns_analysis.get("rnorm100", 0.0) / 100.0

    # 提取Sharpe
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0)
    if sharpe_ratio is None:
        sharpe_ratio = 0.0

    # 提取回撤
    max_drawdown = drawdown_analysis.get("max", {})
    max_dd_pct = max_drawdown.get("drawdown", 0.0) / 100.0 if max_drawdown else 0.0
    max_dd_duration = max_drawdown.get("len", 0) if max_drawdown else 0

    # 提取交易统计
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won_trades = trade_analysis.get("won", {}).get("total", 0)
    lost_trades = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0.0

    # 盈亏比
    avg_win = trade_analysis.get("won", {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_loss = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0)
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

    # 计算Sortino和Calmar
    # 简化处理 - 使用收益序列计算
    trade_returns = [t["pnlcomm"] / initial_cash for t in trade_collector["trades"]]
    if trade_returns:
        returns_series = pd.Series(trade_returns)
        downside_returns = returns_series[returns_series < 0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = float(returns_series.mean() / downside_std) if downside_std > 0 else 0.0

        volatility = float(returns_series.std())
        calmar_ratio = annual_return / max_dd_pct if max_dd_pct > 0 else 0.0
    else:
        sortino_ratio = 0.0
        volatility = 0.0
        calmar_ratio = 0.0

    results_dict = {
        "initial_cash": initial_cash,
        "final_value": float(final_value),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_dd_pct),
        "max_drawdown_duration": int(max_dd_duration),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "calmar_ratio": float(calmar_ratio),
        "volatility": float(volatility),
        "total_trades": int(total_trades),
        "won_trades": int(won_trades),
        "lost_trades": int(lost_trades),
        "trades": trade_collector["trades"],
        "params": params,
    }

    logger.info(
        f"回测完成: 总收益={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, "
        f"最大回撤={max_dd_pct:.2%}, 交易次数={total_trades}"
    )

    return results_dict
