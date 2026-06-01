"""
FastAPI application for AI Strategy Generator.

Provides REST API endpoints for strategy generation, optimization,
template management, and health checks.
"""

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backtester import BacktestResult
from data_fetcher import fetch_stock_daily, generate_sample_data
from optimizer import OptimizationResult, grid_search
from strategy_agent import StrategyAgent, StrategySpec
from strategy_templates import list_templates, render_template, get_template

app = FastAPI(
    title="AI Strategy Generator",
    description="Generate, optimize, and backtest trading strategies using AI",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request to generate and backtest a strategy from natural language."""

    description: str = Field(description="Natural language strategy description")
    symbol: str = Field(default="000001", description="Stock symbol for backtesting")
    days: int = Field(default=365, description="Number of days of historical data")
    initial_cash: float = Field(default=100000, description="Starting cash amount")


class GenerateResponse(BaseModel):
    """Response from strategy generation."""

    spec: StrategySpec
    code: str
    is_valid: bool
    validation_msg: str
    result: BacktestResult | None = None


class OptimizeRequest(BaseModel):
    """Request to optimize strategy parameters."""

    strategy_code: str = Field(description="Python code for the Backtrader strategy")
    symbol: str = Field(default="000001", description="Stock symbol")
    days: int = Field(default=365, description="Number of days of historical data")
    param_grid: dict[str, list[Any]] | None = Field(
        default=None, description="Custom param grid (uses defaults if None)"
    )
    strategy_type: str | None = Field(
        default=None, description="Strategy type for suggested params (used if param_grid is None)"
    )
    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    initial_cash: float = Field(default=100000, description="Starting cash")


class TemplateRunRequest(BaseModel):
    """Request to run a pre-built template strategy."""

    params: dict[str, Any] | None = Field(default=None, description="Template parameters")
    symbol: str = Field(default="000001", description="Stock symbol")
    days: int = Field(default=365, description="Number of days of historical data")
    initial_cash: float = Field(default=100000, description="Starting cash")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-strategy-generator"}


@app.post("/generate", response_model=GenerateResponse)
def generate_strategy(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a trading strategy from a natural language description.

    Uses AI to parse the description, generate Backtrader code,
    validate it, and run a backtest.
    """
    try:
        data = fetch_stock_daily(request.symbol, request.days)
    except Exception:
        data = generate_sample_data(request.symbol, request.days)

    agent = StrategyAgent()
    pipeline_result = agent.run_pipeline(
        user_input=request.description,
        data=data,
        initial_cash=request.initial_cash,
    )

    return GenerateResponse(
        spec=pipeline_result["spec"],
        code=pipeline_result["code"],
        is_valid=pipeline_result["is_valid"],
        validation_msg=pipeline_result["validation_msg"],
        result=pipeline_result["result"],
    )


@app.post("/optimize", response_model=OptimizeResult)
def optimize_strategy(request: OptimizeRequest) -> OptimizationResult:
    """
    Optimize strategy parameters via grid search.

    Accepts strategy code and either a custom parameter grid
    or a strategy type for suggested defaults.
    """
    try:
        data = fetch_stock_daily(request.symbol, request.days)
    except Exception:
        data = generate_sample_data(request.symbol, request.days)

    # Determine param grid
    param_grid = request.param_grid
    if param_grid is None:
        if request.strategy_type is None:
            raise HTTPException(
                status_code=400,
                detail="Either param_grid or strategy_type must be provided",
            )
        param_grid = suggest_params(request.strategy_type)

    try:
        result = grid_search(
            strategy_code=request.strategy_code,
            data=data,
            param_grid=param_grid,
            metric=request.metric,
            initial_cash=request.initial_cash,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/templates")
def get_templates() -> list[dict[str, str]]:
    """List all available strategy templates."""
    return list_templates()


@app.post("/templates/{name}/run", response_model=BacktestResult)
def run_template(name: str, request: TemplateRunRequest) -> BacktestResult:
    """
    Run a pre-built strategy template with optional custom parameters.

    Args:
        name: Template name (e.g., 'ma_crossover', 'rsi_extreme')
    """
    try:
        code = render_template(name, request.params)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        data = fetch_stock_daily(request.symbol, request.days)
    except Exception:
        data = generate_sample_data(request.symbol, request.days)

    from backtester import run_backtest

    result = run_backtest(
        data=data,
        strategy_code=code,
        initial_cash=request.initial_cash,
    )
    return result
