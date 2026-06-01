"""Report formatting utilities."""

import json

from .evaluator import RATING_EMOJI
from .models import DiagnosticReport


def format_report(report: DiagnosticReport, format: str = "text") -> str:
    """Format a diagnostic report.

    Args:
        report: DiagnosticReport to format.
        format: Output format — 'text', 'markdown', or 'json'.

    Returns:
        Formatted report string.
    """
    if format == "json":
        return json.dumps(report.model_dump(), indent=2, default=str)
    elif format == "markdown":
        return _format_markdown(report)
    else:
        return _format_text(report)


def _format_text(report: DiagnosticReport) -> str:
    """Format as plain text with emoji indicators."""
    lines = []
    lines.append("=" * 60)
    lines.append("       STRATEGY DIAGNOSTIC REPORT")
    lines.append("=" * 60)
    lines.append(f"\n📊 Overall Score: {report.overall_score:.1f}/100\n")

    lines.append("─── Metrics ───")
    for name, metric in report.metrics.items():
        emoji = RATING_EMOJI.get(metric.rating, "")
        lines.append(f"  {emoji} {name}: {metric.value:.4f} [{metric.rating.upper()}]")
        lines.append(f"     {metric.explanation}")

    if report.strengths:
        lines.append("\n─── Strengths ───")
        for s in report.strengths:
            lines.append(f"  ✅ {s}")

    if report.weaknesses:
        lines.append("\n─── Weaknesses ───")
        for w in report.weaknesses:
            lines.append(f"  🔴 {w}")

    if report.suggestions:
        lines.append("\n─── Suggestions ───")
        for i, s in enumerate(report.suggestions, 1):
            lines.append(f"  {i}. {s}")

    if report.risk_warnings:
        lines.append("\n─── Risk Warnings ───")
        for w in report.risk_warnings:
            lines.append(f"  🚨 {w}")

    if report.ai_analysis:
        lines.append("\n─── AI Analysis ───")
        lines.append(report.ai_analysis)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _format_markdown(report: DiagnosticReport) -> str:
    """Format as markdown with tables."""
    lines = []
    lines.append("# Strategy Diagnostic Report\n")
    lines.append(f"**Overall Score:** {report.overall_score:.1f}/100\n")

    lines.append("## Metrics\n")
    lines.append("| Metric | Value | Rating |")
    lines.append("|--------|-------|--------|")
    for name, metric in report.metrics.items():
        emoji = RATING_EMOJI.get(metric.rating, "")
        lines.append(f"| {name} | {metric.value:.4f} | {emoji} {metric.rating} |")

    if report.strengths:
        lines.append("\n## Strengths\n")
        for s in report.strengths:
            lines.append(f"- ✅ {s}")

    if report.weaknesses:
        lines.append("\n## Weaknesses\n")
        for w in report.weaknesses:
            lines.append(f"- 🔴 {w}")

    if report.suggestions:
        lines.append("\n## Suggestions\n")
        for i, s in enumerate(report.suggestions, 1):
            lines.append(f"{i}. {s}")

    if report.risk_warnings:
        lines.append("\n## ⚠️ Risk Warnings\n")
        for w in report.risk_warnings:
            lines.append(f"- 🚨 {w}")

    if report.ai_analysis:
        lines.append("\n## AI Analysis\n")
        lines.append(report.ai_analysis)

    return "\n".join(lines)


def format_comparison(reports: list[tuple[str, DiagnosticReport]]) -> str:
    """Format a side-by-side comparison table.

    Args:
        reports: List of (name, DiagnosticReport) tuples.

    Returns:
        Formatted comparison string.
    """
    if not reports:
        return "No strategies to compare."

    lines = []
    lines.append("=" * 80)
    lines.append("       STRATEGY COMPARISON")
    lines.append("=" * 80)

    # Header
    names = [name for name, _ in reports]
    header = f"{'Metric':<20}"
    for name in names:
        header += f" {name:>15}"
    lines.append(header)
    lines.append("-" * (20 + 16 * len(names)))

    # Overall score row
    row = f"{'Overall Score':<20}"
    for _, r in reports:
        row += f" {r.overall_score:>14.1f}"
    lines.append(row)

    # Metric rows
    all_metric_names = set()
    for _, r in reports:
        all_metric_names.update(r.metrics.keys())

    for metric_name in sorted(all_metric_names):
        emoji_row = f"{metric_name:<20}"
        for _, r in reports:
            if metric_name in r.metrics:
                m = r.metrics[metric_name]
                e = RATING_EMOJI.get(m.rating, "")
                emoji_row += f" {e}{m.value:>13.4f}"
            else:
                emoji_row += f" {'N/A':>15}"
        lines.append(emoji_row)

    # Rating rows
    lines.append("")
    rating_row = f"{'Rating':<20}"
    for _, r in reports:
        if reports.index((_, r)) is not None:
            pass
    for metric_name in sorted(all_metric_names):
        rating_row = f"{metric_name + ' Rating':<20}"
        for _, r in reports:
            if metric_name in r.metrics:
                rating_row += f" {r.metrics[metric_name].rating:>15}"
            else:
                rating_row += f" {'N/A':>15}"
        lines.append(rating_row)

    # Best strategy
    best_name, best_report = max(reports, key=lambda x: x[1].overall_score)
    lines.append(f"\n🏆 Recommended: {best_name} (Score: {best_report.overall_score:.1f})")
    lines.append("=" * 80)

    return "\n".join(lines)
