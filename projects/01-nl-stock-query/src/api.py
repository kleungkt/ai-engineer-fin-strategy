"""
FastAPI REST API service for the Natural Language Stock Query system.

Exposes the full query → parse → screen → backtest → diagnostics pipeline
over HTTP endpoints.  Designed for both interactive use and programmatic
integration.

Endpoints
---------
POST /query          – Natural-language stock screening
POST /backtest       – Run a backtest described in plain language
GET  /patterns/{symbol} – Detect K-line patterns for a given symbol
GET  /health         – Liveness / readiness probe

Usage::

    # Start the server
    python api.py
    # or
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

    # Example requests
    curl -X POST http://localhost:8000/query \\
         -H 'Content-Type: application/json' \\
         -d '{"query": "MACD golden cross RSI below 30", "market": "A股", "days": 120}'

    curl -X POST http://localhost:8000/backtest \\
         -H 'Content-Type: application/json' \\
         -d '{"query": "MACD golden cross", "symbol": "AAPL", "days": 250}'

    curl http://localhost:8000/patterns/AAPL?days=60
    curl http://localhost:8000/health
"""

from __future__ import annotations

import logging
import sys
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the Python path so local modules resolve.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from parser import QueryIntent, parse_query  # noqa: E402
from screener import (  # noqa: E402
    ScreenResult,
    generate_sample_data,
    screen_stocks,
)

# ---------------------------------------------------------------------------
# Optional sibling modules – gracefully degrade if they are not yet present.
# ---------------------------------------------------------------------------
try:
    from data_fetcher import fetch_stock_list, fetch_ohlcv  # noqa: E402
    _HAS_DATA_FETCHER = True
except ImportError:
    _HAS_DATA_FETCHER = False

try:
    from backtester import (  # noqa: E402
        BacktestResult,
        map_intent_to_strategy,
        run_backtest,
    )
    _HAS_BACKTESTER = True
except ImportError:
    _HAS_BACKTESTER = False

try:
    from diagnostics import DiagnosticReport, run_diagnostics  # noqa: E402
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False

try:
    from patterns import detect_patterns, summarize_patterns  # noqa: E402
    _HAS_PATTERNS = True
except ImportError:
    _HAS_PATTERNS = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
VERSION = "0.1.0"
APP_TITLE = "Natural Language Stock Query API"
APP_DESCRIPTION = """\
REST API for natural-language stock screening, backtesting, and K-line
pattern detection.  Powered by LLM-based query parsing and pandas-based
technical analysis.
"""

# Default demo stock universe (used when a real data source is unavailable)
DEMO_STOCKS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "V", "MA",
    "JNJ", "PFE", "UNH", "KO", "PEP",
]

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(
        ...,
        description="Natural-language stock screening question",
        examples=["MACD golden cross and RSI below 30"],
    )
    market: str = Field(
        default="A股",
        description="Market scope: A股, 美股, 港股, 加密貨幣, etc.",
    )
    days: int = Field(
        default=120,
        description="Number of historical trading days to consider",
        ge=5,
        le=500,
    )


class IndicatorConditionSchema(BaseModel):
    """Compact representation of a single indicator condition."""

    name: str
    comparison: str
    value: Optional[float] = None
    params: Dict[str, Any] = {}


class QueryIntentSchema(BaseModel):
    """Serializable representation of :class:`parser.QueryIntent`."""

    intent_type: str
    indicators: List[IndicatorConditionSchema]
    stock_scope: str = "A股"
    time_range: Optional[int] = None


class QueryResponse(BaseModel):
    """Response body for ``POST /query``."""

    query_intent: QueryIntentSchema
    matched_stocks: List[Dict[str, Any]]
    total_scanned: int


class BacktestRequest(BaseModel):
    """Request body for ``POST /backtest``."""

    query: str = Field(
        ...,
        description="Natural-language strategy description",
        examples=["MACD golden cross"],
    )
    symbol: str = Field(
        ...,
        description="Stock ticker symbol to backtest on",
        examples=["AAPL"],
    )
    days: int = Field(
        default=250,
        description="Historical trading days for backtest",
        ge=30,
        le=1000,
    )
    initial_cash: float = Field(
        default=100_000.0,
        description="Starting portfolio cash",
        gt=0,
    )


class BacktestResponse(BaseModel):
    """Response body for ``POST /backtest``."""

    strategy_type: str
    params: Dict[str, Any]
    backtest: Dict[str, Any]
    diagnostic: Dict[str, Any]


class PatternResponse(BaseModel):
    """Response body for ``GET /patterns/{symbol}``."""

    symbol: str
    patterns: Dict[str, bool]
    summary: str


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = "ok"
    version: str
    timestamp: str


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=VERSION,
)

# CORS – allow all origins for development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent_to_schema(intent: QueryIntent) -> QueryIntentSchema:
    """Convert a :class:`parser.QueryIntent` to a serializable schema."""
    return QueryIntentSchema(
        intent_type=intent.intent_type.value,
        indicators=[
            IndicatorConditionSchema(
                name=c.name,
                comparison=c.comparison,
                value=c.value,
                params=c.params,
            )
            for c in intent.indicators
        ],
        stock_scope=intent.stock_scope,
        time_range=intent.time_range,
    )


def _resolve_stock_data(
    market: str,
    days: int,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return ``{symbol: DataFrame}`` mapping.

    Uses the real data fetcher if available, otherwise falls back to
    synthetic sample data generated by ``screener.generate_sample_data``.
    """
    if _HAS_DATA_FETCHER:
        try:
            stock_list = fetch_stock_list(market) if symbols is None else symbols
            return {sym: fetch_ohlcv(sym, days=days) for sym in stock_list}
        except Exception as exc:
            logger.warning("Data fetcher failed (%s), falling back to sample data", exc)

    # Fallback: synthetic sample data
    syms = symbols or DEMO_STOCKS
    return {sym: generate_sample_data(sym, days=days) for sym in syms}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query_stocks(req: QueryRequest) -> QueryResponse:
    """Natural-language stock screening.

    Parse a free-form query into structured indicator conditions, fetch
    (or simulate) stock data, and run the screener against all symbols
    in the target market.

    **Request body**:

    * ``query`` – Plain-language description of what you are looking for
      (e.g. *"MACD golden cross and RSI below 30"*).
    * ``market`` – Market scope (default ``A股``).
    * ``days`` – How many trading days of history to evaluate (default 120).

    **Returns**: parsed intent, list of per-stock screening results, and
    total number of stocks scanned.
    """
    # 1. Parse the natural-language query
    try:
        intent: QueryIntent = parse_query(req.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse query: {exc}",
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected parse error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected parse error: {exc}")

    # 2. Fetch stock data
    stock_data = _resolve_stock_data(req.market, req.days)

    # 3. Screen
    results: List[ScreenResult] = screen_stocks(intent, stock_data)

    return QueryResponse(
        query_intent=_intent_to_schema(intent),
        matched_stocks=[r.model_dump() for r in results],
        total_scanned=len(results),
    )


@app.post("/backtest", response_model=BacktestResponse)
async def backtest_strategy(req: BacktestRequest) -> BacktestResponse:
    """Run a backtest described in natural language.

    Parse the user's strategy description, map it to a concrete backtest
    configuration, fetch historical data for the requested symbol, run
    the backtest, and produce a diagnostic report.

    **Request body**:

    * ``query`` – Strategy description in plain language
      (e.g. *"MACD golden cross"*).
    * ``symbol`` – Ticker symbol (e.g. ``AAPL``).
    * ``days`` – Number of trading days for the backtest window (default 250).
    * ``initial_cash`` – Starting capital (default 100 000).

    **Returns**: strategy metadata, backtest results (equity curve, trades,
    performance metrics), and a diagnostic report.
    """
    if not _HAS_BACKTESTER:
        raise HTTPException(
            status_code=501,
            detail=(
                "Backtest engine is not yet available. "
                "Please implement backtester.py and diagnostics.py."
            ),
        )

    # 1. Parse the natural-language query into a strategy intent
    try:
        intent: QueryIntent = parse_query(req.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse query: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")
    except Exception as exc:
        logger.error("Unexpected parse error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected parse error: {exc}")

    # 2. Map the parsed intent to concrete strategy parameters
    try:
        strategy_type, params = map_intent_to_strategy(intent)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot map query to a backtest strategy: {exc}",
        )

    # 3. Fetch historical data for the target symbol
    stock_data = _resolve_stock_data("A股", req.days, symbols=[req.symbol])
    df = stock_data.get(req.symbol)
    if df is None or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient data for symbol '{req.symbol}'",
        )

    # 4. Run the backtest
    try:
        bt_result: BacktestResult = run_backtest(
            df, strategy_type, params, initial_cash=req.initial_cash
        )
    except Exception as exc:
        logger.error("Backtest execution error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    # 5. Run diagnostics
    diag: Dict[str, Any] = {}
    if _HAS_DIAGNOSTICS:
        try:
            diag = run_diagnostics(bt_result).model_dump()
        except Exception as exc:
            logger.warning("Diagnostics failed: %s", exc)
            diag = {"error": str(exc)}
    else:
        diag = {"message": "diagnostics.py not available"}

    return BacktestResponse(
        strategy_type=strategy_type,
        params=params if isinstance(params, dict) else {},
        backtest=bt_result.model_dump() if hasattr(bt_result, "model_dump") else {},
        diagnostic=diag,
    )


@app.get("/patterns/{symbol}", response_model=PatternResponse)
async def get_patterns(
    symbol: str,
    days: int = QueryParam(default=60, ge=5, le=500, description="Look-back window in days"),
) -> PatternResponse:
    """Detect K-line (candlestick) patterns for a given symbol.

    Fetches historical OHLCV data and scans for common candlestick
    patterns such as doji, hammer, engulfing, morning/evening star, etc.

    **Path parameter**:

    * ``symbol`` – Ticker symbol (e.g. ``AAPL``).

    **Query parameter**:

    * ``days`` – Number of historical days to scan (default 60).

    **Returns**: a boolean map of pattern name → detected, plus a
    human-readable summary.
    """
    if not _HAS_PATTERNS:
        # Fallback: provide a basic pattern hint even without the full module
        stock_data = _resolve_stock_data("A股", days, symbols=[symbol])
        df = stock_data.get(symbol)
        if df is None or len(df) < 5:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol '{symbol}'",
            )
        # Minimal candlestick heuristics
        latest = df.iloc[-1]
        body = abs(float(latest["close"]) - float(latest["open"]))
        full_range = float(latest["high"]) - float(latest["low"])
        is_doji = full_range > 0 and (body / full_range) < 0.1
        patterns_map: Dict[str, bool] = {
            "doji": is_doji,
            "hammer": False,
            "engulfing": False,
            "morning_star": False,
            "evening_star": False,
        }
        summary_parts = [k for k, v in patterns_map.items() if v]
        summary = (
            f"Detected: {', '.join(summary_parts)}" if summary_parts
            else "No prominent patterns detected (basic analysis only; install patterns.py for full detection)."
        )
        return PatternResponse(symbol=symbol, patterns=patterns_map, summary=summary)

    # Full pattern detection
    stock_data = _resolve_stock_data("A股", days, symbols=[symbol])
    df = stock_data.get(symbol)
    if df is None or len(df) < 5:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for symbol '{symbol}'",
        )

    try:
        patterns_map = detect_patterns(df)
    except Exception as exc:
        logger.error("Pattern detection error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {exc}")

    try:
        summary = summarize_patterns(patterns_map)
    except Exception:
        summary_parts = [k for k, v in patterns_map.items() if v]
        summary = f"Detected: {', '.join(summary_parts)}" if summary_parts else "No patterns detected."

    return PatternResponse(symbol=symbol, patterns=patterns_map, summary=summary)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe.

    Returns service status, the current API version, and an ISO-8601
    UTC timestamp.
    """
    return HealthResponse(
        status="ok",
        version=VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
