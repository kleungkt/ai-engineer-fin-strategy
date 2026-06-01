"""
Walk-Forward validation module.
前向验证模块 - 通过滚动窗口验证策略的稳健性，防止过拟合。
"""

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WalkForwardResult(BaseModel):
    """Result of walk-forward validation."""

    fold_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results for each fold (IS and OOS metrics)"
    )
    avg_oos_return: float = Field(0.0, description="Average out-of-sample return across all folds")
    avg_is_return: float = Field(0.0, description="Average in-sample return across all folds")
    is_robust: bool = Field(False, description="Whether strategy passes robustness check")
    degradation_ratio: float = Field(
        0.0, description="Ratio of OOS performance to IS performance (higher is better)"
    )


def _split_data_fold(
    data: pd.DataFrame, fold_idx: int, n_splits: int, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    将数据分割为训练集和测试集。

    采用扩展窗口方式：每个fold的训练窗口逐步扩大，测试窗口紧随其后。
    """
    n = len(data)
    # 计算每个fold的总大小
    fold_size = n // n_splits
    # 训练集结束位置
    total_fold_size = int(fold_size / (1 - train_ratio))
    # 起始位置
    start = fold_idx * fold_size
    # 训练结束位置
    train_end = start + int(total_fold_size * train_ratio)
    # 测试结束位置
    test_end = min(start + total_fold_size, n)

    # 确保不超过数据范围
    if train_end >= n:
        train_end = n - 1
    if test_end > n:
        test_end = n
    if train_end >= test_end:
        train_end = test_end - 1

    train_data = data.iloc[start:train_end].copy()
    test_data = data.iloc[train_end:test_end].copy()

    return train_data, test_data


def _calculate_return(results: dict[str, Any]) -> float:
    """从回测结果中提取收益率。"""
    return float(results.get("total_return", 0.0))


def walk_forward_validate(
    data: pd.DataFrame,
    strategy_fn: Callable[..., dict[str, Any]],
    params: dict[str, Any],
    n_splits: int = 5,
    train_ratio: float = 0.7,
) -> WalkForwardResult:
    """
    Walk-forward validation to assess strategy robustness.

    Splits data into n_splits folds. For each fold, runs the strategy
    on both in-sample (training) and out-of-sample (testing) data,
    then compares performance to detect overfitting.

    Args:
        data: OHLCV price data
        strategy_fn: Strategy function that accepts (data, **params) and returns results dict
        params: Strategy parameters to validate
        n_splits: Number of walk-forward folds
        train_ratio: Fraction of each fold used for training

    Returns:
        WalkForwardResult with per-fold results and robustness assessment
    """
    logger.info(f"前向验证: {n_splits} 个fold, 训练比例: {train_ratio}")

    fold_results: list[dict[str, Any]] = []
    is_returns: list[float] = []
    oos_returns: list[float] = []

    for fold_idx in range(n_splits):
        train_data, test_data = _split_data_fold(data, fold_idx, n_splits, train_ratio)

        if len(train_data) < 30 or len(test_data) < 10:
            logger.warning(f"Fold {fold_idx}: 数据不足，跳过 (train={len(train_data)}, test={len(test_data)})")
            continue

        try:
            # 样本内回测
            is_result = strategy_fn(train_data, **params)
            is_return = _calculate_return(is_result)

            # 样本外回测
            oos_result = strategy_fn(test_data, **params)
            oos_return = _calculate_return(oos_result)

            fold_record = {
                "fold": fold_idx,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "is_return": is_return,
                "oos_return": oos_return,
                "is_metrics": is_result,
                "oos_metrics": oos_result,
            }
            fold_results.append(fold_record)
            is_returns.append(is_return)
            oos_returns.append(oos_return)

            logger.info(
                f"Fold {fold_idx}: IS收益={is_return:.4f}, OOS收益={oos_return:.4f}"
            )

        except Exception as e:
            logger.error(f"Fold {fold_idx} 执行失败: {e}")
            fold_results.append({
                "fold": fold_idx,
                "error": str(e),
                "is_return": 0.0,
                "oos_return": 0.0,
            })

    # 计算汇总指标
    avg_is_return = float(np.mean(is_returns)) if is_returns else 0.0
    avg_oos_return = float(np.mean(oos_returns)) if oos_returns else 0.0

    # 计算退化比率 (OOS/IS)
    if avg_is_return != 0:
        degradation_ratio = avg_oos_return / avg_is_return
    else:
        degradation_ratio = 0.0 if avg_oos_return != 0 else 1.0

    # 判断策略是否稳健：OOS/IS > 0.5
    is_robust = degradation_ratio > 0.5 and avg_oos_return > 0

    logger.info(
        f"前向验证完成: 平均IS={avg_is_return:.4f}, 平均OOS={avg_oos_return:.4f}, "
        f"退化比率={degradation_ratio:.4f}, 稳健={is_robust}"
    )

    return WalkForwardResult(
        fold_results=fold_results,
        avg_oos_return=avg_oos_return,
        avg_is_return=avg_is_return,
        is_robust=is_robust,
        degradation_ratio=degradation_ratio,
    )
