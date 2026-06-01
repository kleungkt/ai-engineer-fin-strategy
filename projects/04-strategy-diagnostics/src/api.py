"""FastAPI endpoints for strategy diagnostics."""

from fastapi import FastAPI
from pydantic import BaseModel

from .evaluator import evaluate_backtest
from .formatter import format_comparison, format_report
from .models import BacktestResult, DiagnosticReport

app = FastAPI(title="Strategy Diagnostics API", version="1.0.0")


class CompareRequest(BaseModel):
    """Request body for strategy comparison."""

    strategies: list[BacktestResult]
    names: list[str] | None = None


class CompareResponse(BaseModel):
    """Response for strategy comparison."""

    reports: list[DiagnosticReport]
    comparison: str


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=DiagnosticReport)
def analyze(result: BacktestResult) -> DiagnosticReport:
    """Analyze a single backtest result.

    Args:
        result: BacktestResult with performance metrics.

    Returns:
        DiagnosticReport with full diagnostics.
    """
    return evaluate_backtest(result)


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    """Compare multiple backtest results.

    Args:
        request: CompareRequest with list of BacktestResults.

    Returns:
        CompareResponse with individual reports and comparison text.
    """
    reports = [evaluate_backtest(s) for s in request.strategies]

    names = request.names or [f"Strategy {i+1}" for i in range(len(reports))]
    comparison_input = list(zip(names, reports))
    comparison_text = format_comparison(comparison_input)

    return CompareResponse(reports=reports, comparison=comparison_text)
