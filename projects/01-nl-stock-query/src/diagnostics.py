"""
Strategy diagnostics and AI-powered suggestions.

Evaluates backtest results, generates diagnostic reports with scores,
strengths/weaknesses, and AI-powered professional analysis.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from backtester import BacktestResult


class MetricSummary(BaseModel):
    """Summary for a single metric."""

    value: float
    rating: str  # 'good', 'acceptable', 'poor', 'critical'


class DiagnosticReport(BaseModel):
    """Comprehensive diagnostic report for a backtest result."""

    overall_score: float = Field(ge=0, le=100, description="Overall score 0-100")
    metrics_summary: dict[str, MetricSummary]
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    ai_analysis: str = ""


def _rate_sharpe(sharpe: float) -> str:
    if sharpe > 1.5:
        return "good"
    elif sharpe > 0.5:
        return "acceptable"
    elif sharpe >= 0:
        return "poor"
    else:
        return "critical"


def _rate_drawdown(drawdown: float) -> str:
    dd = abs(drawdown)
    if dd < 10:
        return "good"
    elif dd < 20:
        return "acceptable"
    elif dd < 30:
        return "poor"
    else:
        return "critical"


def _rate_win_rate(win_rate: float) -> str:
    if win_rate > 50:
        return "good"
    elif win_rate > 40:
        return "acceptable"
    else:
        return "poor"


def _rate_return(annual_return: float) -> str:
    if annual_return > 20:
        return "good"
    elif annual_return > 5:
        return "acceptable"
    elif annual_return > 0:
        return "poor"
    else:
        return "critical"


def _score_sharpe(sharpe: float) -> float:
    """Score 0-30 based on Sharpe ratio."""
    return max(0.0, min(30.0, sharpe / 2.0 * 30.0))


def _score_return(annual_return: float) -> float:
    """Score 0-25 based on annual return."""
    return max(0.0, min(25.0, annual_return / 30.0 * 25.0))


def _score_drawdown(max_drawdown: float) -> float:
    """Score 0-25 based on max drawdown (lower is better)."""
    dd = abs(max_drawdown)
    score = max(0.0, min(25.0, (1.0 - dd / 50.0) * 25.0))
    return score


def _score_win_rate(win_rate: float) -> float:
    """Score 0-20 based on win rate."""
    return max(0.0, min(20.0, win_rate / 100.0 * 20.0))


def evaluate_backtest(result: BacktestResult) -> DiagnosticReport:
    """Evaluate a backtest result and produce a diagnostic report.

    Calculates a weighted score from Sharpe ratio, annual return,
    max drawdown, and win rate. Identifies strengths, weaknesses,
    and generates rule-based improvement suggestions.
    """
    # Score calculation
    score_sharpe = _score_sharpe(result.sharpe_ratio)
    score_return = _score_return(result.annual_return)
    score_dd = _score_drawdown(result.max_drawdown)
    score_wr = _score_win_rate(result.win_rate)
    overall_score = round(score_sharpe + score_return + score_dd + score_wr, 1)

    # Metrics summary
    metrics: dict[str, MetricSummary] = {
        "sharpe_ratio": MetricSummary(value=result.sharpe_ratio, rating=_rate_sharpe(result.sharpe_ratio)),
        "annual_return": MetricSummary(value=result.annual_return, rating=_rate_return(result.annual_return)),
        "max_drawdown": MetricSummary(value=result.max_drawdown, rating=_rate_drawdown(result.max_drawdown)),
        "win_rate": MetricSummary(value=result.win_rate, rating=_rate_win_rate(result.win_rate)),
        "total_trades": MetricSummary(value=float(result.total_trades), rating="acceptable" if result.total_trades >= 30 else "poor"),
    }

    # Strengths and weaknesses
    strengths: list[str] = []
    weaknesses: list[str] = []

    if _rate_sharpe(result.sharpe_ratio) == "good":
        strengths.append(f"Excellent risk-adjusted returns (Sharpe {result.sharpe_ratio:.2f})")
    elif _rate_sharpe(result.sharpe_ratio) in ("poor", "critical"):
        weaknesses.append(f"Poor risk-adjusted returns (Sharpe {result.sharpe_ratio:.2f})")

    if _rate_return(result.annual_return) == "good":
        strengths.append(f"Strong annual return of {result.annual_return:.1f}%")
    elif _rate_return(result.annual_return) in ("poor", "critical"):
        weaknesses.append(f"Weak annual return of {result.annual_return:.1f}%")

    if _rate_drawdown(result.max_drawdown) == "good":
        strengths.append(f"Controlled drawdown ({abs(result.max_drawdown):.1f}%)")
    elif _rate_drawdown(result.max_drawdown) in ("poor", "critical"):
        weaknesses.append(f"Excessive drawdown ({abs(result.max_drawdown):.1f}%)")

    if _rate_win_rate(result.win_rate) == "good":
        strengths.append(f"High win rate ({result.win_rate:.1f}%)")
    elif _rate_win_rate(result.win_rate) == "poor":
        weaknesses.append(f"Low win rate ({result.win_rate:.1f}%)")

    if result.total_trades < 30:
        weaknesses.append(f"Low trade count ({result.total_trades}) may reduce statistical significance")

    # Rule-based suggestions
    suggestions: list[str] = []

    if abs(result.max_drawdown) >= 20:
        suggestions.append("Drawdown too high — consider adding stop-loss or trailing stop mechanisms")
    if abs(result.max_drawdown) >= 30:
        suggestions.append("Critical drawdown level — reduce position sizing or add risk limits")

    if result.win_rate < 40:
        suggestions.append("Win rate below 40% — review entry signals or add confirmation filters")

    if result.sharpe_ratio < 0.5:
        suggestions.append("Low Sharpe ratio — strategy may not be compensating for risk adequately")

    if result.annual_return < 5:
        suggestions.append("Low annual return — consider optimizing parameters or adding alpha sources")

    if result.total_trades < 30:
        suggestions.append("Low trade count — extend backtest period or consider more instruments")

    if result.max_drawdown_duration > 180:
        suggestions.append(f"Long drawdown recovery ({result.max_drawdown_duration} days) — consider portfolio diversification")

    if not suggestions:
        suggestions.append("Strategy looks solid. Consider stress-testing across different market regimes.")

    return DiagnosticReport(
        overall_score=overall_score,
        metrics_summary=metrics,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )


def generate_ai_analysis(
    report: DiagnosticReport,
    strategy_type: str,
    params: dict[str, Any],
) -> str:
    """Generate a professional AI-powered strategy analysis using OpenAI.

    Args:
        report: The diagnostic report with metrics.
        strategy_type: Description of the strategy type (e.g., 'momentum', 'mean-reversion').
        params: Strategy parameters used.

    Returns:
        A detailed professional analysis string.
    """
    client = OpenAI()

    metrics_text = "\n".join(
        f"- {name}: {m.value:.2f} ({m.rating})"
        for name, m in report.metrics_summary.items()
    )

    prompt = f"""You are a senior quantitative analyst. Analyze the following backtest results
and provide a professional strategy assessment report.

## Strategy Type: {strategy_type}
## Parameters: {params}

## Metrics:
{metrics_text}

## Overall Score: {report.overall_score}/100
## Strengths: {', '.join(report.strengths) if report.strengths else 'None identified'}
## Weaknesses: {', '.join(report.weaknesses) if report.weaknesses else 'None identified'}
## Suggestions: {', '.join(report.suggestions)}

Please provide:
1. **Overall Assessment** — Is this strategy viable? How does it compare to a benchmark?
2. **Risk Warnings** — Key risks and scenarios where this strategy could underperform.
3. **Improvement Suggestions** — Concrete, actionable recommendations to improve performance.

Write in a professional, analytical tone suitable for an investment committee.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quantitative finance expert providing strategy analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    return response.choices[0].message.content or ""


def format_report(report: DiagnosticReport) -> str:
    """Format a DiagnosticReport as a readable text report with emoji indicators.

    Args:
        report: The diagnostic report to format.

    Returns:
        Formatted string report.
    """
    rating_emoji = {"good": "✅", "acceptable": "⚠️", "poor": "🔴", "critical": "🚨"}

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("📊  STRATEGY DIAGNOSTIC REPORT")
    lines.append("=" * 60)
    lines.append(f"\n🎯 Overall Score: {report.overall_score:.1f} / 100")
    lines.append("")

    lines.append("📈 Metrics Summary:")
    lines.append("-" * 40)
    for name, metric in report.metrics_summary.items():
        emoji = rating_emoji.get(metric.rating, "❓")
        lines.append(f"  {emoji} {name}: {metric.value:.2f} [{metric.rating.upper()}]")
    lines.append("")

    if report.strengths:
        lines.append("💪 Strengths:")
        for s in report.strengths:
            lines.append(f"  ✅ {s}")
        lines.append("")

    if report.weaknesses:
        lines.append("⚠️  Weaknesses:")
        for w in report.weaknesses:
            lines.append(f"  🔴 {w}")
        lines.append("")

    if report.suggestions:
        lines.append("💡 Suggestions:")
        for s in report.suggestions:
            lines.append(f"  → {s}")
        lines.append("")

    if report.ai_analysis:
        lines.append("🤖 AI Analysis:")
        lines.append("-" * 40)
        lines.append(report.ai_analysis)
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
