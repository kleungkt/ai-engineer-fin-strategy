"""Pydantic models for the Strategy Diagnostics system."""

from typing import Optional

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    """Backtest result data."""

    total_return: float = Field(description="Total return as decimal (e.g. 0.5 for 50%)")
    annual_return: float = Field(description="Annualized return as decimal")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown as positive decimal (e.g. 0.15 for 15%)")
    max_drawdown_duration: int = Field(description="Max drawdown duration in days")
    win_rate: float = Field(description="Win rate as decimal (e.g. 0.55 for 55%)")
    total_trades: int = Field(description="Total number of trades")
    trade_log: list[dict] = Field(default_factory=list, description="Detailed trade log")
    equity_curve: Optional[list[float]] = Field(default=None, description="Equity curve values")


class MetricRating(BaseModel):
    """Rating for a single metric."""

    value: float = Field(description="The metric value")
    rating: str = Field(description="Rating: excellent/good/acceptable/poor/critical")
    explanation: str = Field(description="Human-readable explanation")


class DiagnosticReport(BaseModel):
    """Full diagnostic report for a backtest."""

    overall_score: float = Field(ge=0, le=100, description="Overall score 0-100")
    metrics: dict[str, MetricRating] = Field(description="Individual metric ratings")
    strengths: list[str] = Field(default_factory=list, description="Strategy strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Strategy weaknesses")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    risk_warnings: list[str] = Field(default_factory=list, description="Risk warnings for critical metrics")
    ai_analysis: str = Field(default="", description="AI-generated analysis text")
