"""
Overfit detection module.
过拟合检测模块 - 通过多维度分析判断策略是否过拟合。
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OverfitReport(BaseModel):
    """Report on overfitting likelihood for a strategy."""

    overfit_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overfit probability score (0=no overfit, 1=certain overfit)"
    )
    warnings: list[str] = Field(default_factory=list, description="Overfitting warning messages")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations to reduce overfitting")


def _check_sharpe_too_high(is_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查Sharpe比率是否异常高。"""
    sharpe = float(is_results.get("sharpe_ratio", 0.0))
    if sharpe > 5.0:
        return 0.4, f"Sharpe比率异常高 ({sharpe:.2f})，超过5.0，可能过拟合"
    elif sharpe > 3.0:
        return 0.2, f"Sharpe比率较高 ({sharpe:.2f})，超过3.0，需谨慎验证"
    return 0.0, None


def _check_performance_cliff(is_results: dict[str, Any], oos_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查样本内外是否存在性能断崖。"""
    is_return = float(is_results.get("total_return", 0.0))
    oos_return = float(oos_results.get("total_return", 0.0))

    if is_return > 0 and oos_return < 0:
        # 样本内盈利但样本外亏损 - 严重警告
        gap = is_return - oos_return
        if gap > 0.3:
            return 0.4, f"性能断崖: 样本内收益 {is_return:.2%} vs 样本外收益 {oos_return:.2%}，差距 {gap:.2%}"
        return 0.25, f"样本内盈利但样本外亏损: IS={is_return:.2%}, OOS={oos_return:.2%}"

    # Sharpe比率大幅下降
    is_sharpe = float(is_results.get("sharpe_ratio", 0.0))
    oos_sharpe = float(oos_results.get("sharpe_ratio", 0.0))
    if is_sharpe > 1.0 and oos_sharpe < 0:
        return 0.3, f"Sharpe比率断崖: IS={is_sharpe:.2f} vs OOS={oos_sharpe:.2f}"

    return 0.0, None


def _check_drawdown_increase(is_results: dict[str, Any], oos_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查样本外最大回撤是否大幅增加。"""
    is_dd = abs(float(is_results.get("max_drawdown", 0.0)))
    oos_dd = abs(float(oos_results.get("max_drawdown", 0.0)))

    if is_dd > 0 and oos_dd > is_dd * 2.5:
        return 0.2, f"样本外回撤大幅增加: IS最大回撤 {is_dd:.2%} vs OOS {oos_dd:.2%}"
    elif is_dd > 0 and oos_dd > is_dd * 1.8:
        return 0.1, f"样本外回撤增加: IS最大回撤 {is_dd:.2%} vs OOS {oos_dd:.2%}"

    return 0.0, None


def _check_win_rate_drop(is_results: dict[str, Any], oos_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查胜率是否大幅下降。"""
    is_wr = float(is_results.get("win_rate", 0.0))
    oos_wr = float(oos_results.get("win_rate", 0.0))

    if is_wr > 0.6 and oos_wr < 0.4:
        return 0.15, f"胜率大幅下降: IS={is_wr:.2%} vs OOS={oos_wr:.2%}"

    return 0.0, None


def _check_excessive_params(is_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查参数数量是否过多。"""
    # 尝试从结果中获取参数数量信息
    params = is_results.get("params", {})
    n_params = len(params) if isinstance(params, dict) else 0

    if n_params > 10:
        return 0.2, f"策略参数过多 ({n_params}个)，增加过拟合风险"
    elif n_params > 6:
        return 0.1, f"策略参数较多 ({n_params}个)，建议精简"

    return 0.0, None


def _check_consistency(is_results: dict[str, Any], oos_results: dict[str, Any]) -> tuple[float, str | None]:
    """检查各指标的一致性。"""
    is_profit_factor = float(is_results.get("profit_factor", 0.0))
    oos_profit_factor = float(oos_results.get("profit_factor", 0.0))

    if is_profit_factor > 2.0 and oos_profit_factor < 1.0:
        return 0.15, f"盈亏比断崖: IS={is_profit_factor:.2f} vs OOS={oos_profit_factor:.2f}"

    return 0.0, None


def _generate_recommendations(
    is_results: dict[str, Any],
    oos_results: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    """根据检测结果生成改进建议。"""
    recommendations: list[str] = []

    is_sharpe = float(is_results.get("sharpe_ratio", 0.0))
    oos_sharpe = float(oos_results.get("sharpe_ratio", 0.0))

    if is_sharpe > 3.0:
        recommendations.append("降低对Sharpe比率的期望，真实市场中2.0以上已属优秀")

    if any("性能断崖" in w for w in warnings):
        recommendations.append("使用前向验证 (Walk-Forward) 重新评估策略稳健性")
        recommendations.append("减少策略参数数量，使用更简单的逻辑")

    if any("回撤" in w for w in warnings):
        recommendations.append("增加风险管理规则，如动态止损")
        recommendations.append("在更多历史数据上测试策略，覆盖不同市场环境")

    if any("参数过多" in w for w in warnings):
        recommendations.append("精简策略逻辑，保留核心交易信号")
        recommendations.append("使用参数优化器的随机搜索代替网格搜索")

    if any("Sharpe" in w for w in warnings):
        recommendations.append("检查是否使用了未来数据（look-ahead bias）")
        recommendations.append("考虑交易成本和滑点的影响")

    if any("胜率" in w for w in warnings):
        recommendations.append("避免过度优化入场条件")
        recommendations.append("增加样本外验证数据量")

    if not recommendations:
        recommendations.append("策略表现稳健，建议继续监控实盘表现")

    return recommendations


def detect_overfit(is_results: dict[str, Any], oos_results: dict[str, Any]) -> OverfitReport:
    """
    Detect if a strategy is likely overfitted.

    Compares in-sample and out-of-sample metrics across multiple dimensions
    to calculate an overfit probability score.

    Args:
        is_results: In-sample (training) backtest results
        oos_results: Out-of-sample (testing) backtest results

    Returns:
        OverfitReport with score, warnings, and recommendations
    """
    logger.info("开始过拟合检测...")

    warnings: list[str] = []
    total_score = 0.0

    # 检查各项指标
    checks = [
        _check_sharpe_too_high(is_results),
        _check_performance_cliff(is_results, oos_results),
        _check_drawdown_increase(is_results, oos_results),
        _check_win_rate_drop(is_results, oos_results),
        _check_excessive_params(is_results),
        _check_consistency(is_results, oos_results),
    ]

    for score, warning in checks:
        total_score += score
        if warning:
            warnings.append(warning)

    # 限制分数在 0-1 范围内
    overfit_score = min(max(total_score, 0.0), 1.0)

    # 生成建议
    recommendations = _generate_recommendations(is_results, oos_results, warnings)

    if not warnings:
        warnings.append("未检测到明显的过拟合迹象")

    logger.info(f"过拟合检测完成: 得分={overfit_score:.2f}, 警告数={len(warnings)}")

    return OverfitReport(
        overfit_score=overfit_score,
        warnings=warnings,
        recommendations=recommendations,
    )
