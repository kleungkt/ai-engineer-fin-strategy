"""
Backtest module for AI Strategy Generator.
回测模块 - 提供策略回测引擎、性能指标计算和报告生成功能。
"""

from .engine import run_strategy_backtest
from .metrics import calculate_metrics
from .report_generator import generate_report

__all__ = [
    "run_strategy_backtest",
    "calculate_metrics",
    "generate_report",
]
