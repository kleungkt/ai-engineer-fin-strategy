"""AI-powered strategy analysis using OpenAI."""

from openai import OpenAI

from .models import DiagnosticReport


def _build_analysis_prompt(
    report: DiagnosticReport,
    strategy_name: str = "",
    params: dict | None = None,
) -> str:
    """Build the prompt for AI analysis."""
    metrics_text = ""
    for name, metric in report.metrics.items():
        metrics_text += f"  - {name}: value={metric.value:.4f}, rating={metric.rating}\n"

    params_text = ""
    if params:
        params_text = "\nStrategy Parameters:\n"
        for k, v in params.items():
            params_text += f"  - {k}: {v}\n"

    return f"""You are a senior buy-side quantitative analyst. Analyze the following backtest results
and write a professional strategy diagnostic report.

Strategy Name: {strategy_name or "Unnamed Strategy"}
{params_text}

Performance Metrics:
{metrics_text}
Overall Score: {report.overall_score}/100

Strengths:
{chr(10).join(f"  - {s}" for s in report.strengths) if report.strengths else "  None identified"}

Weaknesses:
{chr(10).join(f"  - {w}" for w in report.weaknesses) if report.weaknesses else "  None identified"}

Risk Warnings:
{chr(10).join(f"  - {w}" for w in report.risk_warnings) if report.risk_warnings else "  None"}

Please provide:
1. **Overall Assessment**: A concise summary of strategy quality and viability.
2. **Risk Profile**: Analysis of risk characteristics and tail risk exposure.
3. **Improvement Suggestions**: Specific, actionable recommendations to improve performance.
4. **Market Condition Suitability**: Which market regimes this strategy may perform well/poorly in.

Write in a professional, concise tone suitable for an investment committee presentation."""


def generate_analysis(
    report: DiagnosticReport,
    strategy_name: str = "",
    params: dict | None = None,
) -> str:
    """Generate an AI-powered analysis of a diagnostic report.

    Args:
        report: DiagnosticReport from evaluator.
        strategy_name: Optional name of the strategy.
        params: Optional strategy parameters.

    Returns:
        Professional analysis text.
    """
    client = OpenAI()
    prompt = _build_analysis_prompt(report, strategy_name, params)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior quantitative analyst at a top hedge fund."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2000,
    )

    return response.choices[0].message.content


def generate_comparison(reports: list[tuple[str, DiagnosticReport]]) -> str:
    """Compare multiple strategies and recommend the best one.

    Args:
        reports: List of (strategy_name, DiagnosticReport) tuples.

    Returns:
        Professional comparison analysis text.
    """
    strategies_text = ""
    for name, report in reports:
        metrics_text = ""
        for metric_name, metric in report.metrics.items():
            metrics_text += f"    - {metric_name}: {metric.value:.4f} ({metric.rating})\n"
        strategies_text += f"""
Strategy: {name}
  Overall Score: {report.overall_score}/100
  Metrics:
{metrics_text}
  Strengths: {', '.join(report.strengths) if report.strengths else 'None'}
  Weaknesses: {', '.join(report.weaknesses) if report.weaknesses else 'None'}
"""

    prompt = f"""You are a senior portfolio manager. Compare the following strategies and provide:
1. A side-by-side comparison of key metrics
2. Relative strengths and weaknesses of each
3. A clear recommendation on which strategy to deploy and why
4. Suggestions for portfolio allocation if combining strategies

Strategies:
{strategies_text}

Provide a concise, professional analysis."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior portfolio manager at a quantitative hedge fund."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2500,
    )

    return response.choices[0].message.content
