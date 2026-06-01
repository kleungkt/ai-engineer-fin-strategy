"""
Parameter optimization module.
参数优化模块 - 提供网格搜索和随机搜索两种优化方法。
"""

import itertools
import random
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    """Result of a parameter optimization run."""

    best_params: dict[str, Any] = Field(..., description="Best parameter combination found")
    best_score: float = Field(..., description="Score achieved by best parameters")
    all_results: list[dict[str, Any]] = Field(
        default_factory=list, description="All parameter combinations and their scores"
    )
    method: str = Field(..., description="Optimization method used (grid_search or random_search)")


def _extract_metric(results: dict[str, Any], metric: str) -> float:
    """从回测结果中提取指定指标值。"""
    metric_map = {
        "sharpe": "sharpe_ratio",
        "sharpe_ratio": "sharpe_ratio",
        "return": "total_return",
        "total_return": "total_return",
        "annual_return": "annual_return",
        "sortino": "sortino_ratio",
        "sortino_ratio": "sortino_ratio",
        "calmar": "calmar_ratio",
        "calmar_ratio": "calmar_ratio",
        "max_drawdown": "max_drawdown",
        "win_rate": "win_rate",
    }
    key = metric_map.get(metric.lower(), metric)
    value = results.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _generate_param_combinations(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """生成所有参数组合。"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def _sample_random_params(param_ranges: dict[str, Any]) -> dict[str, Any]:
    """从参数范围中随机采样一组参数。"""
    params = {}
    for key, range_spec in param_ranges.items():
        if isinstance(range_spec, (list, tuple)):
            if len(range_spec) == 2:
                low, high = range_spec
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = random.randint(low, high)
                else:
                    params[key] = random.uniform(low, high)
            else:
                # 离散值列表，随机选择
                params[key] = random.choice(range_spec)
        elif isinstance(range_spec, dict):
            # 支持 {"type": "int/float/choice", "low": ..., "high": ..., "values": [...]} 格式
            rtype = range_spec.get("type", "float")
            if rtype == "choice":
                params[key] = random.choice(range_spec["values"])
            elif rtype == "int":
                params[key] = random.randint(range_spec["low"], range_spec["high"])
            else:
                params[key] = random.uniform(range_spec["low"], range_spec["high"])
        else:
            # 固定值
            params[key] = range_spec
    return params


def grid_search(
    data: pd.DataFrame,
    strategy_fn: Callable[..., dict[str, Any]],
    param_grid: dict[str, list],
    metric: str = "sharpe",
) -> OptimizationResult:
    """
    Exhaustive grid search over all parameter combinations.

    Args:
        data: OHLCV price data
        strategy_fn: Strategy function that accepts (data, **params) and returns results dict
        param_grid: Dict mapping parameter names to lists of values to try
        metric: Metric to optimize (sharpe, return, sortino, etc.)

    Returns:
        OptimizationResult with best parameters and all trial results
    """
    combinations = _generate_param_combinations(param_grid)
    logger.info(f"网格搜索: 共 {len(combinations)} 种参数组合")

    all_results: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}

    for i, params in enumerate(combinations):
        try:
            result = strategy_fn(data, **params)
            score = _extract_metric(result, metric)
            trial_record = {"params": params, "score": score, "metrics": result}
            all_results.append(trial_record)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if (i + 1) % 10 == 0:
                logger.info(f"网格搜索进度: {i + 1}/{len(combinations)}, 当前最优: {best_score:.4f}")

        except Exception as e:
            logger.warning(f"参数组合 {params} 执行失败: {e}")
            all_results.append({"params": params, "score": 0.0, "error": str(e)})

    logger.info(f"网格搜索完成. 最优参数: {best_params}, 最优得分: {best_score:.4f}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=all_results,
        method="grid_search",
    )


def random_search(
    data: pd.DataFrame,
    strategy_fn: Callable[..., dict[str, Any]],
    param_ranges: dict[str, Any],
    n_trials: int = 100,
    metric: str = "sharpe",
) -> OptimizationResult:
    """
    Random search over parameter space.

    More efficient than grid search for high-dimensional parameter spaces.

    Args:
        data: OHLCV price data
        strategy_fn: Strategy function that accepts (data, **params) and returns results dict
        param_ranges: Dict mapping parameter names to [low, high] ranges or value lists
        n_trials: Number of random trials to run
        metric: Metric to optimize (sharpe, return, sortino, etc.)

    Returns:
        OptimizationResult with best parameters and all trial results
    """
    logger.info(f"随机搜索: 共 {n_trials} 次试验")

    all_results: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}

    for i in range(n_trials):
        params = _sample_random_params(param_ranges)
        try:
            result = strategy_fn(data, **params)
            score = _extract_metric(result, metric)
            trial_record = {"params": params, "score": score, "metrics": result}
            all_results.append(trial_record)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if (i + 1) % 20 == 0:
                logger.info(f"随机搜索进度: {i + 1}/{n_trials}, 当前最优: {best_score:.4f}")

        except Exception as e:
            logger.warning(f"参数 {params} 执行失败: {e}")
            all_results.append({"params": params, "score": 0.0, "error": str(e)})

    logger.info(f"随机搜索完成. 最优参数: {best_params}, 最优得分: {best_score:.4f}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=all_results,
        method="random_search",
    )
