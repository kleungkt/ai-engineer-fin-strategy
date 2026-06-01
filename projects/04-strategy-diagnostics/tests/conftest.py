"""Shared test fixtures."""

import pytest

from src.models import BacktestResult, DiagnosticReport, MetricRating


@pytest.fixture
def excellent_backtest() -> BacktestResult:
    """A backtest with excellent metrics."""
    return BacktestResult(
        total_return=1.5,
        annual_return=0.45,
        sharpe_ratio=2.5,
        max_drawdown=0.03,
        max_drawdown_duration=10,
        win_rate=0.65,
        total_trades=100,
        trade_log=[{"entry": 100, "exit": 105, "pnl": 5} for _ in range(100)],
        equity_curve=[100 + i * 0.5 for i in range(200)],
    )


@pytest.fixture
def mediocre_backtest() -> BacktestResult:
    """A backtest with mediocre/acceptable metrics."""
    return BacktestResult(
        total_return=0.15,
        annual_return=0.08,
        sharpe_ratio=0.6,
        max_drawdown=0.18,
        max_drawdown_duration=60,
        win_rate=0.42,
        total_trades=80,
        trade_log=[{"entry": 100, "exit": 101, "pnl": 1} for _ in range(80)],
    )


@pytest.fixture
def poor_backtest() -> BacktestResult:
    """A backtest with poor/critical metrics."""
    return BacktestResult(
        total_return=-0.2,
        annual_return=-0.1,
        sharpe_ratio=-0.5,
        max_drawdown=0.45,
        max_drawdown_duration=200,
        win_rate=0.25,
        total_trades=50,
        trade_log=[{"entry": 100, "exit": 99, "pnl": -1} for _ in range(50)],
    )


@pytest.fixture
def sample_report() -> DiagnosticReport:
    """A sample diagnostic report."""
    return DiagnosticReport(
        overall_score=72.5,
        metrics={
            "sharpe_ratio": MetricRating(value=1.6, rating="good", explanation="Good Sharpe"),
            "annual_return": MetricRating(value=0.20, rating="good", explanation="Good return"),
            "max_drawdown": MetricRating(value=0.12, rating="acceptable", explanation="Acceptable DD"),
            "win_rate": MetricRating(value=0.55, rating="good", explanation="Good win rate"),
        },
        strengths=["Sharpe ratio is good"],
        weaknesses=[],
        suggestions=["Consider tighter stops"],
        risk_warnings=[],
    )
