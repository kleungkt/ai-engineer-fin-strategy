"""
Unified Pipeline — FastAPI
==========================
REST API for the E2E pipeline.
"""

from __future__ import annotations

import sys
import time
from typing import Any

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from unified_pipeline import UnifiedPipeline, PipelineResult

app = FastAPI(
    title="AI Strategy Pipeline API",
    description="E2E pipeline: NL query → intent parsing → stock screening → strategy generation → backtest → diagnosis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Pipeline 分析請求"""
    nl_request: str = Field(..., description="自然語言請求", example="帮我找最近 RSI 低於 30 的 A 股")
    llm_model: str = Field(default="gpt-4o-mini", description="使用的 LLM 模型")
    demo_mode: bool = Field(default=True, description="演示模式（不調用真實 API）")
    max_stocks: int = Field(default=10, ge=1, le=50, description="最多返回股票數")


class StockResultResponse(BaseModel):
    symbol: str
    name: str | None
    matched_conditions: list[str]


class BacktestResultResponse(BaseModel):
    strategy_name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


class DiagnosticResultResponse(BaseModel):
    score: int
    rating: str
    metrics_rating: dict[str, str]
    ai_analysis: str | None


class PipelineResponse(BaseModel):
    """完整管線響應"""
    request: str
    execution_time_ms: float
    stocks_found: int
    strategies_generated: int
    summary: str
    intent: dict[str, Any]
    stocks: list[StockResultResponse]
    backtest_results: list[BacktestResultResponse]
    diagnostics: list[DiagnosticResultResponse]


# ── Endpoints ────────────────────────────────────────────────


@app.post("/pipeline/analyze", response_model=PipelineResponse)
def analyze(request: AnalyzeRequest):
    """執行完整 E2E 管線"""
    start = time.perf_counter()

    pipeline = UnifiedPipeline(
        llm_model=request.llm_model,
        max_stocks=request.max_stocks,
        demo_mode=request.demo_mode,
    )

    try:
        result: PipelineResult = pipeline.run(request.nl_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PipelineResponse(
        request=result.original_request,
        execution_time_ms=round(elapsed_ms, 1),
        stocks_found=len(result.matched_stocks),
        strategies_generated=len(result.strategy_results),
        summary=result.summary,
        intent=result.parsed_intent,
        stocks=[
            StockResultResponse(
                symbol=s.symbol,
                name=s.name,
                matched_conditions=s.matched_conditions,
            )
            for s in result.matched_stocks
        ],
        backtest_results=[
            BacktestResultResponse(
                strategy_name=sr.backtest_result.get("strategy_name", "Unknown")
                if sr.backtest_result
                else "N/A",
                total_return=sr.backtest_result.get("total_return", 0)
                if sr.backtest_result
                else 0,
                annual_return=sr.backtest_result.get("annual_return", 0)
                if sr.backtest_result
                else 0,
                sharpe_ratio=sr.backtest_result.get("sharpe_ratio", 0)
                if sr.backtest_result
                else 0,
                max_drawdown=sr.backtest_result.get("max_drawdown", 0)
                if sr.backtest_result
                else 0,
                win_rate=sr.backtest_result.get("win_rate", 0)
                if sr.backtest_result
                else 0,
                total_trades=sr.backtest_result.get("total_trades", 0)
                if sr.backtest_result
                else 0,
            )
            for sr in result.strategy_results
        ],
        diagnostics=[
            DiagnosticResultResponse(
                score=d.score,
                rating=d.rating,
                metrics_rating=d.metrics_rating,
                ai_analysis=d.ai_analysis,
            )
            for d in result.diagnostic_results
        ],
    )


@app.get("/pipeline/health")
def health():
    return {"status": "ok", "service": "pipeline-api"}