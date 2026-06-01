"""
Extended performance metrics module.
扩展性能指标模块 - 计算全面的策略评估指标。
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 中国市场无风险利率 (年化约3%)
RISK_FREE_RATE = 0.03
# 年化交易日数
TRADING_DAYS_PER_YEAR = 244


def calculate_metrics(returns: pd.Series) -> dict[str, Any]:
    """
    Calculate comprehensive performance metrics from a returns series.

    Computes standard quantitative metrics for strategy evaluation including
    return metrics, risk metrics, and risk-adjusted metrics.

    Args:
        returns: Series of periodic (daily) returns (e.g., 0.01 for 1%)

    Returns:
        Dictionary of metric name to value
    """
    if returns.empty:
        logger.warning("空收益序列，返回默认指标")
        return _default_metrics()

    # 确保为float类型
    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        return _default_metrics()

    # === 收益指标 ===
    total_return = float((1 + returns).prod() - 1)
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    annual_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1) if total_return > -1 else -1.0

    # === 风险指标 ===
    volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    # 最大回撤持续期
    max_drawdown_duration = _calculate_max_drawdown_duration(drawdowns)

    # === 风险调整指标 ===
    # Sharpe比率
    excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe_ratio = float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ) if excess_returns.std() > 0 else 0.0

    # Sortino比率
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside_returns) > 0 else 0.0001
    sortino_ratio = float((annual_return - RISK_FREE_RATE) / downside_std) if downside_std > 0 else 0.0

    # Calmar比率
    calmar_ratio = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # === 交易指标 ===
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    win_rate = float(len(positive_returns) / len(returns)) if len(returns) > 0 else 0.0

    # 盈亏比
    avg_gain = float(positive_returns.mean()) if len(positive_returns) > 0 else 0.0
    avg_loss = float(abs(negative_returns.mean())) if len(negative_returns) > 0 else 0.0001
    profit_factor = float(avg_gain / avg_loss) if avg_loss > 0 else 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "calmar_ratio": calmar_ratio,
        "volatility": volatility,
    }


def _calculate_max_drawdown_duration(drawdowns: pd.Series) -> int:
    """计算最大回撤持续期（交易日数）。"""
    is_in_drawdown = drawdowns < 0
    if not is_in_drawdown.any():
        return 0

    # 找到所有回撤区间
    groups = (~is_in_drawdown).cumsum()
    drawdown_periods = is_in_drawdown.groupby(groups).sum()

    return int(drawdown_periods.max()) if len(drawdown_periods) > 0 else 0


def _default_metrics() -> dict[str, Any]:
    """返回默认的空指标。"""
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "calmar_ratio": 0.0,
        "volatility": 0.0,
    }
