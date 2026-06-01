"""
Report generator module.
报告生成器模块 - 生成人类可读的策略回测报告。
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _format_pct(value: float) -> str:
    """格式化百分比。"""
    return f"{value:.2%}"


def _format_float(value: float, decimals: int = 2) -> str:
    """格式化浮点数。"""
    return f"{value:.{decimals}f}"


def _format_money(value: float) -> str:
    """格式化金额。"""
    return f"¥{value:,.2f}"


def _generate_text_report(results: dict[str, Any]) -> str:
    """生成纯文本格式报告。"""
    lines: list[str] = []
    separator = "=" * 60
    thin_sep = "-" * 60

    # 标题
    lines.append(separator)
    lines.append("              策略回测报告 (Strategy Backtest Report)")
    lines.append(separator)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # === 摘要指标 ===
    lines.append("【收益概览】")
    lines.append(thin_sep)
    initial_cash = results.get("initial_cash", 100000)
    final_value = results.get("final_value", initial_cash)
    lines.append(f"  初始资金:       {_format_money(initial_cash)}")
    lines.append(f"  最终净值:       {_format_money(final_value)}")
    lines.append(f"  总收益率:       {_format_pct(results.get('total_return', 0.0))}")
    lines.append(f"  年化收益率:     {_format_pct(results.get('annual_return', 0.0))}")
    lines.append("")

    # === 风险指标 ===
    lines.append("【风险指标】")
    lines.append(thin_sep)
    lines.append(f"  最大回撤:       {_format_pct(results.get('max_drawdown', 0.0))}")
    lines.append(f"  回撤持续期:     {results.get('max_drawdown_duration', 0)} 交易日")
    lines.append(f"  年化波动率:     {_format_pct(results.get('volatility', 0.0))}")
    lines.append("")

    # === 风险调整指标 ===
    lines.append("【风险调整指标】")
    lines.append(thin_sep)
    lines.append(f"  Sharpe比率:     {_format_float(results.get('sharpe_ratio', 0.0))}")
    lines.append(f"  Sortino比率:    {_format_float(results.get('sortino_ratio', 0.0))}")
    lines.append(f"  Calmar比率:     {_format_float(results.get('calmar_ratio', 0.0))}")
    lines.append("")

    # === 交易统计 ===
    lines.append("【交易统计】")
    lines.append(thin_sep)
    total_trades = results.get("total_trades", 0)
    won = results.get("won_trades", 0)
    lost = results.get("lost_trades", 0)
    lines.append(f"  总交易次数:     {total_trades}")
    lines.append(f"  盈利交易:       {won}")
    lines.append(f"  亏损交易:       {lost}")
    lines.append(f"  胜率:           {_format_pct(results.get('win_rate', 0.0))}")
    lines.append(f"  盈亏比:         {_format_float(results.get('profit_factor', 0.0))}")
    lines.append("")

    # === 交易明细 ===
    trades = results.get("trades", [])
    if trades:
        lines.append("【交易明细】")
        lines.append(thin_sep)
        lines.append(f"  {'入场日期':<12} {'出场日期':<12} {'盈亏':>12} {'盈亏(扣费)':>12}")
        lines.append(f"  {'-'*10:<12} {'-'*10:<12} {'-'*10:>12} {'-'*10:>12}")

        for trade in trades[:20]:  # 最多显示20笔
            entry = trade.get("entry_date", "N/A")
            exit_d = trade.get("exit_date", "N/A")
            pnl = trade.get("pnl", 0.0)
            pnlcomm = trade.get("pnlcomm", 0.0)
            lines.append(f"  {entry:<12} {exit_d:<12} {pnl:>12,.2f} {pnlcomm:>12,.2f}")

        if len(trades) > 20:
            lines.append(f"  ... 共 {len(trades)} 笔交易，仅显示前20笔")
        lines.append("")

    # === 风险分析 ===
    lines.append("【风险分析】")
    lines.append(thin_sep)
    sharpe = results.get("sharpe_ratio", 0.0)
    max_dd = results.get("max_drawdown", 0.0)

    if sharpe >= 2.0:
        lines.append("  ✓ Sharpe比率优秀 (>2.0)")
    elif sharpe >= 1.0:
        lines.append("  ○ Sharpe比率良好 (1.0-2.0)")
    elif sharpe > 0:
        lines.append("  △ Sharpe比率偏低 (0-1.0)")
    else:
        lines.append("  ✗ Sharpe比率为负，策略亏损")

    if abs(max_dd) < 0.1:
        lines.append("  ✓ 最大回撤可控 (<10%)")
    elif abs(max_dd) < 0.2:
        lines.append("  ○ 最大回撤中等 (10%-20%)")
    else:
        lines.append("  △ 最大回撤较大 (>20%)，注意风险控制")

    win_rate = results.get("win_rate", 0.0)
    if win_rate >= 0.6:
        lines.append("  ✓ 胜率较高 (≥60%)")
    elif win_rate >= 0.4:
        lines.append("  ○ 胜率正常 (40%-60%)")
    else:
        lines.append("  △ 胜率偏低 (<40%)，需配合高盈亏比")
    lines.append("")

    # === 参数 ===
    params = results.get("params", {})
    if params:
        lines.append("【策略参数】")
        lines.append(thin_sep)
        for key, value in params.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # === 错误信息 ===
    error = results.get("error")
    if error:
        lines.append("【错误信息】")
        lines.append(thin_sep)
        lines.append(f"  {error}")
        lines.append("")

    lines.append(separator)
    lines.append("报告结束")
    lines.append(separator)

    return "\n".join(lines)


def _generate_markdown_report(results: dict[str, Any]) -> str:
    """生成Markdown格式报告。"""
    lines: list[str] = []

    lines.append("# 策略回测报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 收益概览
    lines.append("## 收益概览\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    initial_cash = results.get("initial_cash", 100000)
    final_value = results.get("final_value", initial_cash)
    lines.append(f"| 初始资金 | {_format_money(initial_cash)} |")
    lines.append(f"| 最终净值 | {_format_money(final_value)} |")
    lines.append(f"| 总收益率 | {_format_pct(results.get('total_return', 0.0))} |")
    lines.append(f"| 年化收益率 | {_format_pct(results.get('annual_return', 0.0))} |")
    lines.append("")

    # 风险指标
    lines.append("## 风险指标\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 最大回撤 | {_format_pct(results.get('max_drawdown', 0.0))} |")
    lines.append(f"| 回撤持续期 | {results.get('max_drawdown_duration', 0)} 交易日 |")
    lines.append(f"| 年化波动率 | {_format_pct(results.get('volatility', 0.0))} |")
    lines.append("")

    # 风险调整指标
    lines.append("## 风险调整指标\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| Sharpe比率 | {_format_float(results.get('sharpe_ratio', 0.0))} |")
    lines.append(f"| Sortino比率 | {_format_float(results.get('sortino_ratio', 0.0))} |")
    lines.append(f"| Calmar比率 | {_format_float(results.get('calmar_ratio', 0.0))} |")
    lines.append("")

    # 交易统计
    lines.append("## 交易统计\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总交易次数 | {results.get('total_trades', 0)} |")
    lines.append(f"| 盈利交易 | {results.get('won_trades', 0)} |")
    lines.append(f"| 亏损交易 | {results.get('lost_trades', 0)} |")
    lines.append(f"| 胜率 | {_format_pct(results.get('win_rate', 0.0))} |")
    lines.append(f"| 盈亏比 | {_format_float(results.get('profit_factor', 0.0))} |")
    lines.append("")

    # 交易明细
    trades = results.get("trades", [])
    if trades:
        lines.append("## 交易明细\n")
        lines.append("| 入场日期 | 出场日期 | 盈亏 | 盈亏(扣费) |")
        lines.append("|----------|----------|------|-----------|")
        for trade in trades[:30]:
            entry = trade.get("entry_date", "N/A")
            exit_d = trade.get("exit_date", "N/A")
            pnl = trade.get("pnl", 0.0)
            pnlcomm = trade.get("pnlcomm", 0.0)
            lines.append(f"| {entry} | {exit_d} | {pnl:,.2f} | {pnlcomm:,.2f} |")
        if len(trades) > 30:
            lines.append(f"\n*共 {len(trades)} 笔交易，仅显示前30笔*\n")
        lines.append("")

    # 参数
    params = results.get("params", {})
    if params:
        lines.append("## 策略参数\n")
        for key, value in params.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")

    return "\n".join(lines)


def generate_report(results: dict[str, Any], format: str = "text") -> str:
    """
    Generate a human-readable backtest report.

    Creates a formatted report including summary metrics, trade log,
    risk analysis, and strategy parameters.

    Args:
        results: Backtest results dictionary from run_strategy_backtest
        format: Output format - 'text' for plain text, 'markdown' for Markdown

    Returns:
        Formatted report string
    """
    if format.lower() == "markdown" or format.lower() == "md":
        return _generate_markdown_report(results)
    else:
        return _generate_text_report(results)
