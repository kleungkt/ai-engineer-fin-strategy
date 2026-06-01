"""Tests for the unified E2E pipeline."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the pipeline module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from unified_pipeline import (
    UnifiedPipeline,
    PipelineResult,
    StockResult,
    StrategyResult,
    DiagnosticResult,
)


class TestPipelineResult:
    """Test PipelineResult data class."""

    def test_pipeline_result_to_dict(self):
        result = PipelineResult(
            original_request="test",
            parsed_intent={"intent_type": "screener"},
            matched_stocks=[],
            strategy_results=[],
            diagnostic_results=[],
            summary="test summary",
        )
        d = result.to_dict()
        assert d["request"] == "test"
        assert d["stocks_found"] == 0
        assert d["strategies_generated"] == 0


class TestStockResult:
    """Test StockResult data class."""

    def test_stock_result_creation(self):
        stock = StockResult(
            symbol="000001",
            name="平安银行",
            matched_conditions=["RSI below 30"],
            metrics={"rsi": 25.0, "close": 12.5},
        )
        assert stock.symbol == "000001"
        assert stock.name == "平安银行"
        assert len(stock.matched_conditions) == 1
        assert stock.metrics["rsi"] == 25.0


class TestStrategyResult:
    """Test StrategyResult data class."""

    def test_strategy_result_valid(self):
        result = StrategyResult(
            natural_language="test strategy",
            generated_code="import backtrader as bt\nclass MyStrat(bt.Strategy): pass",
            backtest_result={
                "strategy_name": "MA_Cross",
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "total_trades": 20,
                "annual_return": 0.12,
            },
            is_valid=True,
            validation_errors=[],
            error_message=None,
        )
        assert result.is_valid is True
        assert result.backtest_result["total_return"] == 0.15

    def test_strategy_result_invalid(self):
        result = StrategyResult(
            natural_language="bad strategy",
            generated_code=None,
            backtest_result=None,
            is_valid=False,
            validation_errors=["Syntax error"],
            error_message="Parse failed",
        )
        assert result.is_valid is False
        assert result.backtest_result is None


class TestDiagnosticResult:
    """Test DiagnosticResult data class."""

    def test_diagnostic_result_creation(self):
        diag = DiagnosticResult(
            score=85,
            rating="Excellent",
            metrics_rating={"sharpe_ratio": "excellent", "win_rate": "good"},
            ai_analysis="Strong strategy with good risk-adjusted returns.",
            formatted_report="Overall Score: 85/100\n...",
        )
        assert diag.score == 85
        assert diag.rating == "Excellent"


class TestUnifiedPipelineDemo:
    """Test UnifiedPipeline in demo mode (no API needed)."""

    def test_pipeline_initialization(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        assert pipeline.demo_mode is True
        assert pipeline.max_stocks == 10

    def test_pipeline_intent_parsing(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        intent = pipeline._parse_intent("帮我找 RSI 低於 30 的股票")
        assert "intent_type" in intent
        assert intent["intent_type"] == "indicator_screener"

    def test_pipeline_demo_stock_results(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        intent = {"intent_type": "indicator_screener", "indicators": ["RSI"], "stock_scope": "A股"}
        stocks = pipeline._demo_stock_results(intent)
        assert len(stocks) > 0
        assert all(isinstance(s, StockResult) for s in stocks)

    def test_pipeline_demo_strategy_result(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        strategy = pipeline._demo_strategy_result("布林通道策略")
        assert strategy.is_valid is True
        assert strategy.generated_code is not None
        assert "backtrader" in strategy.generated_code.lower()
        assert strategy.backtest_result is not None

    def test_pipeline_demo_diagnosis(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        strategy = StrategyResult(
            natural_language="test",
            generated_code="import backtrader as bt\nclass S(bt.Strategy): pass",
            backtest_result={
                "strategy_name": "Test",
                "total_return": 0.1,
                "annual_return": 0.08,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.05,
                "win_rate": 0.6,
                "total_trades": 15,
                "max_drawdown_duration": 5,
            },
            is_valid=True,
            validation_errors=[],
            error_message=None,
        )
        diag = pipeline._diagnose(strategy)
        assert diag.score > 0
        assert diag.rating in ("Excellent", "Good", "Acceptable", "Poor", "Critical")
        assert len(diag.metrics_rating) > 0

    def test_pipeline_full_run(self):
        pipeline = UnifiedPipeline(demo_mode=True, max_stocks=5)
        result = pipeline.run("帮我找最近 RSI 低於 30 的 A 股")
        assert isinstance(result, PipelineResult)
        assert result.original_request == "帮我找最近 RSI 低於 30 的 A 股"
        assert len(result.matched_stocks) > 0
        assert len(result.strategy_results) > 0
        assert len(result.diagnostic_results) > 0
        assert result.summary != ""


class TestPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_pipeline_handles_empty_request(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        result = pipeline.run("")
        # Empty request still gets demo intent + stock screening in demo mode
        assert isinstance(result, PipelineResult)
        assert result.original_request == ""

    def test_pipeline_max_stocks_limit(self):
        pipeline = UnifiedPipeline(demo_mode=True, max_stocks=2)
        result = pipeline.run("帮我找 RSI 低於 30 的股票")
        assert len(result.matched_stocks) <= 2

    def test_pipeline_diagnose_no_backtest(self):
        pipeline = UnifiedPipeline(demo_mode=True)
        strategy = StrategyResult(
            natural_language="test",
            generated_code=None,
            backtest_result=None,
            is_valid=False,
            validation_errors=[],
            error_message=None,
        )
        diag = pipeline._diagnose(strategy)
        assert diag.score == 0
        assert diag.rating == "N/A"


class TestGetDemoSymbols:
    """Test demo symbol generation."""

    def test_a_shares_symbols(self):
        from unified_pipeline import UnifiedPipeline

        symbols = UnifiedPipeline._get_demo_symbols("A股", 5)
        assert len(symbols) == 5
        assert "000001" in symbols

    def test_us_stocks_symbols(self):
        from unified_pipeline import UnifiedPipeline

        symbols = UnifiedPipeline._get_demo_symbols("美股", 3)
        assert len(symbols) == 3
        assert "AAPL" in symbols

    def test_default_symbols(self):
        from unified_pipeline import UnifiedPipeline

        symbols = UnifiedPipeline._get_demo_symbols("未知市場", 2)
        assert len(symbols) == 2