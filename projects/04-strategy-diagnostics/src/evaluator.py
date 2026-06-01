"""Backtest evaluator with metric rating and diagnostics."""

from .models import BacktestResult, DiagnosticReport, MetricRating

# Rating thresholds: list of (threshold, rating) ordered for rate_metric
# For higher_is_better: first threshold that value >= wins
# For lower_is_better: first threshold that value <= wins

SHARPE_THRESHOLDS: list[tuple[float, str]] = [
    (2.0, "excellent"),
    (1.5, "good"),
    (0.5, "acceptable"),
    (0.0, "poor"),
    (-float("inf"), "critical"),
]

ANNUAL_RETURN_THRESHOLDS: list[tuple[float, str]] = [
    (0.30, "excellent"),
    (0.15, "good"),
    (0.05, "acceptable"),
    (0.0, "poor"),
    (-float("inf"), "critical"),
]

MAX_DRAWDOWN_THRESHOLDS: list[tuple[float, str]] = [
    (0.05, "excellent"),
    (0.10, "good"),
    (0.20, "acceptable"),
    (0.30, "poor"),
    (float("inf"), "critical"),
]

WIN_RATE_THRESHOLDS: list[tuple[float, str]] = [
    (0.60, "excellent"),
    (0.50, "good"),
    (0.40, "acceptable"),
    (0.30, "poor"),
    (-float("inf"), "critical"),
]

RATING_SCORES: dict[str, float] = {
    "excellent": 100.0,
    "good": 75.0,
    "acceptable": 50.0,
    "poor": 25.0,
    "critical": 0.0,
}

RATING_EMOJI: dict[str, str] = {
    "excellent": "✅",
    "good": "✅",
    "acceptable": "⚠️",
    "poor": "🔴",
    "critical": "🚨",
}

METRIC_WEIGHTS: dict[str, float] = {
    "sharpe_ratio": 0.30,
    "annual_return": 0.25,
    "max_drawdown": 0.25,
    "win_rate": 0.20,
}

METRIC_DISPLAY: dict[str, str] = {
    "sharpe_ratio": "Sharpe Ratio",
    "annual_return": "Annual Return",
    "max_drawdown": "Max Drawdown",
    "win_rate": "Win Rate",
}


def rate_metric(
    value: float,
    thresholds: list[tuple[float, str]],
    lower_is_better: bool = False,
) -> MetricRating:
    """Rate a metric value against thresholds.

    Args:
        value: The metric value to rate.
        thresholds: List of (threshold, rating) tuples.
            For higher_is_better: sorted descending, first match where value >= threshold.
            For lower_is_better: sorted ascending by threshold, first match where value <= threshold.
        lower_is_better: If True, lower values get better ratings.

    Returns:
        MetricRating with value, rating, and explanation.
    """
    if lower_is_better:
        # thresholds should be ascending: (0.05, excellent), (0.10, good), ...
        for threshold, rating in thresholds:
            if value <= threshold:
                explanation = f"Value {value:.4f} is {rating} (threshold: ≤{threshold:.2f})"
                return MetricRating(value=value, rating=rating, explanation=explanation)
        # Should not reach here if thresholds are well-formed
        explanation = f"Value {value:.4f} is critical"
        return MetricRating(value=value, rating="critical", explanation=explanation)
    else:
        # thresholds sorted descending: (2.0, excellent), (1.5, good), ...
        for threshold, rating in thresholds:
            if value >= threshold:
                explanation = f"Value {value:.4f} is {rating} (threshold: ≥{threshold:.2f})"
                return MetricRating(value=value, rating=rating, explanation=explanation)
        explanation = f"Value {value:.4f} is critical"
        return MetricRating(value=value, rating="critical", explanation=explanation)


def _generate_strengths(metrics: dict[str, MetricRating]) -> list[str]:
    """Generate strengths list from metric ratings."""
    strengths = []
    for name, metric in metrics.items():
        display = METRIC_DISPLAY.get(name, name)
        if metric.rating in ("excellent", "good"):
            strengths.append(f"{display} is {metric.rating}: {metric.explanation}")
    return strengths


def _generate_weaknesses(metrics: dict[str, MetricRating]) -> list[str]:
    """Generate weaknesses list from metric ratings."""
    weaknesses = []
    for name, metric in metrics.items():
        display = METRIC_DISPLAY.get(name, name)
        if metric.rating in ("poor", "critical"):
            weaknesses.append(f"{display} is {metric.rating}: {metric.explanation}")
    return weaknesses


def _generate_suggestions(metrics: dict[str, MetricRating], result: BacktestResult) -> list[str]:
    """Generate improvement suggestions based on ratings."""
    suggestions = []

    sharpe = metrics.get("sharpe_ratio")
    if sharpe and sharpe.rating in ("poor", "critical"):
        suggestions.append(
            "Consider reducing position sizing or improving entry signals to boost risk-adjusted returns."
        )

    drawdown = metrics.get("max_drawdown")
    if drawdown and drawdown.rating in ("poor", "critical"):
        suggestions.append(
            "Implement tighter stop-losses or reduce leverage to limit maximum drawdown."
        )

    win_rate = metrics.get("win_rate")
    if win_rate and win_rate.rating in ("poor", "critical"):
        suggestions.append(
            "Review entry criteria — low win rate suggests signals may need refinement or additional filters."
        )

    annual_ret = metrics.get("annual_return")
    if annual_ret and annual_ret.rating in ("poor", "critical"):
        suggestions.append(
            "Strategy returns are weak. Consider adding alpha sources or optimizing parameters."
        )

    if result.total_trades < 30:
        suggestions.append(
            f"Only {result.total_trades} trades — results may lack statistical significance. "
            "Consider extending the backtest period."
        )

    if not suggestions:
        suggestions.append("Strategy performs well across all metrics. Consider live testing with small capital.")

    return suggestions


def _generate_risk_warnings(metrics: dict[str, MetricRating]) -> list[str]:
    """Generate risk warnings for critical metrics."""
    warnings = []
    for name, metric in metrics.items():
        display = METRIC_DISPLAY.get(name, name)
        if metric.rating == "critical":
            warnings.append(f"⚠️ CRITICAL: {display} is at critical level — {metric.explanation}")
    return warnings


def evaluate_backtest(result: BacktestResult) -> DiagnosticReport:
    """Evaluate a backtest result and generate a full diagnostic report.

    Args:
        result: BacktestResult with performance metrics.

    Returns:
        DiagnosticReport with scores, ratings, strengths, weaknesses, suggestions.
    """
    metrics: dict[str, MetricRating] = {}

    metrics["sharpe_ratio"] = rate_metric(result.sharpe_ratio, SHARPE_THRESHOLDS)
    metrics["annual_return"] = rate_metric(result.annual_return, ANNUAL_RETURN_THRESHOLDS)
    metrics["max_drawdown"] = rate_metric(result.max_drawdown, MAX_DRAWDOWN_THRESHOLDS, lower_is_better=True)
    metrics["win_rate"] = rate_metric(result.win_rate, WIN_RATE_THRESHOLDS)

    # Calculate weighted overall score
    total_weight = 0.0
    weighted_score = 0.0
    for name, weight in METRIC_WEIGHTS.items():
        if name in metrics:
            weighted_score += RATING_SCORES[metrics[name].rating] * weight
            total_weight += weight

    overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

    strengths = _generate_strengths(metrics)
    weaknesses = _generate_weaknesses(metrics)
    suggestions = _generate_suggestions(metrics, result)
    risk_warnings = _generate_risk_warnings(metrics)

    return DiagnosticReport(
        overall_score=round(overall_score, 2),
        metrics=metrics,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        risk_warnings=risk_warnings,
    )
