"""
Unified E2E Pipeline — 串聯 P1 + P3 + P4
==========================================
從自然語言查詢 → 意圖解析 → 股票篩選 → 策略生成 → 回測 → 診斷，全鏈路一鍵完成。

Usage:
    from pipeline.unified_pipeline import UnifiedPipeline
    pipeline = UnifiedPipeline()
    result = pipeline.run("帮我找最近 RSI 低於 30 的 A 股")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── P1 imports ────────────────────────────────────────────────
# Use resolve() so __file__ works even when running as `python pipeline/main.py`
_root = Path(__file__).resolve().parent.parent  # pipeline/.. = repo root
sys.path.insert(0, str(_root / "projects" / "01-nl-stock-query" / "src"))
from parser import parse_query
from data_fetcher import fetch_stock_daily
from indicators import (
    calc_ma,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_bollinger,
    calc_kdj,
    check_crossover,
    check_crossunder,
    check_above,
    check_below,
)

# ── P3 imports ────────────────────────────────────────────────
sys.path.insert(0, str(_root / "projects" / "03-ai-strategy-generator" / "src"))
from strategy_agent import StrategyAgent, StrategySpec
from strategy_templates import list_templates, get_template, render_template
from code_validator import validate_python_code, validate_backtrader_strategy

# ── P4 imports (use importlib to load standalone .py files) ──
# P4 files use "from .models" relative imports, so we need to load them
# as separate modules without a package context.
_p4_evaluator = None
_p4_ai_analyst = None
_p4_formatter = None
_p4_models = None


def _get_p4():
    """Lazy-load P4 modules — add parent dir so 'from .models' resolves."""
    global _p4_evaluator, _p4_ai_analyst, _p4_formatter, _p4_models
    if _p4_evaluator is None:
        # Add project dir (above src/) so Python sees "src" as a package
        sys.path.insert(0, str(_root / "projects" / "04-strategy-diagnostics"))
        from src.models import BacktestResult as _BR
        from src.evaluator import evaluate_backtest as _eb
        from src.formatter import format_report as _fr
        from src.ai_analyst import generate_analysis as _ga

        # Store as a dict so we can access by name
        class _P4:
            pass

        m = _P4()
        m.BacktestResult = _BR
        m.evaluate_backtest = _eb
        m.format_report = _fr
        m.generate_analysis = _ga

        _p4_evaluator = m
        _p4_ai_analyst = m
        _p4_formatter = m
        _p4_models = m

    return _p4_evaluator, _p4_ai_analyst, _p4_formatter, _p4_models

# ── Types ─────────────────────────────────────────────────────
from pydantic import BaseModel, Field


@dataclass
class StockResult:
    """單支股票的篩選結果"""
    symbol: str
    name: str | None
    matched_conditions: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """策略生成 + 回測結果"""
    natural_language: str
    generated_code: str | None
    backtest_result: dict[str, Any] | None
    is_valid: bool
    validation_errors: list[str]
    error_message: str | None


@dataclass
class DiagnosticResult:
    """策略診斷結果"""
    score: int
    rating: str
    metrics_rating: dict[str, str]
    ai_analysis: str | None
    formatted_report: str | None


@dataclass
class PipelineResult:
    """完整管線輸出"""
    original_request: str
    parsed_intent: dict[str, Any]
    matched_stocks: list[StockResult]
    strategy_results: list[StrategyResult]
    diagnostic_results: list[DiagnosticResult]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "request": self.original_request,
            "intent": self.parsed_intent,
            "stocks_found": len(self.matched_stocks),
            "strategies_generated": len(self.strategy_results),
            "summary": self.summary,
        }


# ── Core Pipeline ─────────────────────────────────────────────


class UnifiedPipeline:
    """
    統一管線：NL → 意圖解析 → 股票篩選 → 策略生成 → 回測 → 診斷

    Example:
        pipeline = UnifiedPipeline()
        result = pipeline.run("帮我找 MACD 金叉且 RSI < 30 的股票，回测均值回归策略")
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        max_stocks: int = 10,
        demo_mode: bool = False,
    ):
        self.llm_model = llm_model
        self.max_stocks = max_stocks
        self.demo_mode = demo_mode

        self._parser = parse_query  # function, not class
        self._strategy_agent = StrategyAgent(model=llm_model) if not demo_mode else None

    # ── Step 1: 解析 NL ────────────────────────────────────────

    def _parse_intent(self, nl_request: str) -> dict[str, Any]:
        """解析自然語言意圖"""
        if self.demo_mode:
            return {
                "intent_type": "indicator_screener",
                "indicators": ["RSI", "MACD"],
                "stock_scope": "A股",
                "time_range": 30,
            }
        try:
            result = self._parser(nl_request)
            return {
                "intent_type": result.intent_type.value if hasattr(result.intent_type, 'value') else result.intent_type,
                "indicators": [i.model_dump() for i in result.indicators],
                "stock_scope": result.stock_scope,
                "time_range": result.time_range,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Step 2: 股票篩選 ────────────────────────────────────────

    def _screen_stocks(self, intent: dict) -> list[StockResult]:
        """根據意圖篩選股票"""
        if self.demo_mode:
            return self._demo_stock_results(intent)

        indicators = intent.get("indicators", [])
        scope = intent.get("stock_scope", "A股")
        time_range = intent.get("time_range", 30)

        results: list[StockResult] = []

        # Demo symbols for testing
        symbols = self._get_demo_symbols(scope, self.max_stocks)

        for symbol in symbols:
            try:
                df = fetch_stock_daily(symbol, days=time_range + 50)
                if df is None or len(df) < 50:
                    continue

                close = df["close"]
                matched: list[str] = []
                metrics: dict[str, float] = {}

                for ind in indicators:
                    name = ind.get("name", "").upper()
                    comparison = ind.get("comparison", "")
                    value = ind.get("value", 0)

                    if name == "RSI":
                        rsi = calc_rsi(df, period=14)
                        rsi_val = float(rsi.iloc[-1])
                        metrics["rsi"] = rsi_val
                        if comparison == "below" and rsi_val < value:
                            matched.append(f"RSI {comparison} {value} (当前: {rsi_val:.1f})")

                    elif name == "MACD":
                        macd_l, signal_l, _ = calc_macd(df)
                        macd_val = float(macd_l.iloc[-1])
                        signal_val = float(signal_l.iloc[-1])
                        metrics["macd"] = macd_val
                        metrics["macd_signal"] = signal_val
                        if check_crossover(macd_l, signal_l).iloc[-1]:
                            matched.append("MACD 金叉")

                    elif name == "MA":
                        ma_val = float(calc_ma(df, period=20).iloc[-1])
                        metrics["ma20"] = ma_val
                        matched.append(f"MA20 当前: {ma_val:.2f}")

                    elif name == "BOLLINGER":
                        upper, middle, lower = calc_bollinger(df)
                        metrics["bb_upper"] = float(upper.iloc[-1])
                        metrics["bb_middle"] = float(middle.iloc[-1])
                        metrics["bb_lower"] = float(lower.iloc[-1])

                if matched:
                    results.append(
                        StockResult(
                            symbol=symbol,
                            name=symbol,
                            matched_conditions=matched,
                            metrics=metrics,
                        )
                    )
            except Exception:
                continue

        return results[: self.max_stocks]

    def _demo_stock_results(self, intent: dict) -> list[StockResult]:
        """演示模式的虛構股票結果"""
        indicators = intent.get("indicators", [])
        names = ["平安银行", "招商银行", "中国平安", "兴业银行", "宁波银行"]
        symbols = ["000001", "600036", "601318", "601166", "002142"]

        results = []
        import random

        for i in range(min(len(names), self.max_stocks)):
            # indicators may be strings or dicts depending on mode
            if indicators and isinstance(indicators[0], dict):
                matched = [
                    f"{ind.get('name', 'RSI')} {ind.get('comparison', 'below')} {ind.get('value', 30)}"
                    for ind in indicators
                ]
            else:
                matched = [str(ind) for ind in indicators]
            results.append(
                StockResult(
                    symbol=symbols[i],
                    name=names[i],
                    matched_conditions=matched,
                    metrics={
                        "rsi": random.uniform(20, 35),
                        "close": random.uniform(10, 50),
                    },
                )
            )
        return results

    # ── Step 3: 策略生成 + 回測 ────────────────────────────────

    def _generate_and_backtest(
        self, nl_request: str, stock: StockResult | None = None
    ) -> StrategyResult:
        """生成策略代碼並執行回測"""
        if self.demo_mode or self._strategy_agent is None:
            return self._demo_strategy_result(nl_request)

        try:
            # 生成策略
            agent_result = self._strategy_agent.generate(nl_request)

            # 驗證代碼
            is_valid, err = validate_backtrader_strategy(agent_result.code)
            if not is_valid:
                return StrategyResult(
                    natural_language=nl_request,
                    generated_code=agent_result.code,
                    backtest_result=None,
                    is_valid=False,
                    validation_errors=[err],
                    error_message=None,
                )

            # 回測
            sys.path.insert(0, str(_root / "projects" / "03-ai-strategy-generator" / "src"))
            from backtester import run_backtest as p3_run_backtest
            from data_fetcher import generate_sample_data

            data = generate_sample_data(days=100) if self.demo_mode else fetch_stock_daily(
                stock.symbol if stock else "000001", days=100
            )
            bt_result = p3_run_backtest(agent_result.code, data)

            return StrategyResult(
                natural_language=nl_request,
                generated_code=agent_result.code,
                backtest_result={
                    "total_return": bt_result.total_return,
                    "annual_return": bt_result.annual_return,
                    "sharpe_ratio": bt_result.sharpe_ratio,
                    "max_drawdown": bt_result.max_drawdown,
                    "win_rate": bt_result.win_rate,
                    "total_trades": bt_result.total_trades,
                    "strategy_name": bt_result.strategy_name,
                },
                is_valid=True,
                validation_errors=[],
                error_message=None,
            )

        except Exception as e:
            return StrategyResult(
                natural_language=nl_request,
                generated_code=None,
                backtest_result=None,
                is_valid=False,
                validation_errors=[],
                error_message=str(e),
            )

    def _demo_strategy_result(self, nl_request: str) -> StrategyResult:
        """演示模式虛構結果"""
        import random

        return StrategyResult(
            natural_language=nl_request,
            generated_code=self._demo_strategy_code(),
            backtest_result={
                "total_return": random.uniform(-0.05, 0.25),
                "annual_return": random.uniform(0.02, 0.30),
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "max_drawdown": random.uniform(0.05, 0.20),
                "win_rate": random.uniform(0.45, 0.65),
                "total_trades": random.randint(5, 30),
                "strategy_name": "MA_Crossover",
            },
            is_valid=True,
            validation_errors=[],
            error_message=None,
        )

    @staticmethod
    def _demo_strategy_code() -> str:
        return '''import backtrader as bt

class MeanReversion(bt.Strategy):
    params = dict(period=20, std=2)

    def __init__(self):
        sma = bt.ind.SMA(period=self.p.period)
        self.boll = bt.ind.BollingerBands(sma, period=self.p.period, devfactor=self.p.std)

    def next(self):
        if self.data.close < self.boll.lines.bot:
            self.buy()
        elif self.data.close > self.boll.lines.top:
            self.sell()
'''

    # ── Step 4: 診斷 ─────────────────────────────────────────

    def _diagnose(self, strategy_result: StrategyResult) -> DiagnosticResult:
        """對回測結果進行診斷"""
        if strategy_result.backtest_result is None:
            return DiagnosticResult(
                score=0,
                rating="N/A",
                metrics_rating={},
                ai_analysis=None,
                formatted_report="No backtest result available for diagnosis.",
            )

        try:
            ev, ai, fmt, p4models = _get_p4()

            p4_result = p4models.BacktestResult(
                strategy_name=strategy_result.backtest_result.get("strategy_name", "Unknown"),
                total_return=strategy_result.backtest_result.get("total_return", 0),
                annual_return=strategy_result.backtest_result.get("annual_return", 0),
                sharpe_ratio=strategy_result.backtest_result.get("sharpe_ratio", 0),
                max_drawdown=strategy_result.backtest_result.get("max_drawdown", 0),
                max_drawdown_duration=strategy_result.backtest_result.get("max_drawdown_duration", 0),
                win_rate=strategy_result.backtest_result.get("win_rate", 0),
                total_trades=strategy_result.backtest_result.get("total_trades", 0),
            )

            # 評分 → returns DiagnosticReport
            report = ev.evaluate_backtest(p4_result)
            score = int(report.overall_score)
            metrics_rating = {k: v.rating for k, v in report.metrics.items()}

            # Pick overall rating from avg of metrics
            rating_map = {"excellent": "Excellent", "good": "Good", "acceptable": "Acceptable", "poor": "Poor", "critical": "Critical"}
            all_ratings = [v.rating for v in report.metrics.values()]
            # Use avg score to derive rating
            if score >= 80:
                rating = "Excellent"
            elif score >= 70:
                rating = "Good"
            elif score >= 50:
                rating = "Acceptable"
            elif score >= 30:
                rating = "Poor"
            else:
                rating = "Critical"

            # AI 分析（非演示模式）
            ai_analysis = None
            if not self.demo_mode:
                try:
                    ai_report = ai.generate_analysis(p4_result, llm_model=self.llm_model)
                    ai_analysis = ai_report.analysis if hasattr(ai_report, 'analysis') else str(ai_report)
                except Exception:
                    pass

            # 格式化
            formatted = fmt.format_report(report, format="text")

            return DiagnosticResult(
                score=score,
                rating=rating,
                metrics_rating=metrics_rating,
                ai_analysis=ai_analysis,
                formatted_report=formatted,
            )

        except Exception as e:
            return DiagnosticResult(
                score=0,
                rating="ERROR",
                metrics_rating={},
                ai_analysis=None,
                formatted_report=f"Diagnosis error: {str(e)}",
            )

    # ── 主入口 ───────────────────────────────────────────────

    def run(self, nl_request: str) -> PipelineResult:
        """
        執行完整管線

        Args:
            nl_request: 自然語言輸入，如 "帮我找最近 RSI 低於 30 的 A 股"

        Returns:
            PipelineResult: 包含意圖解析、股票篩選、策略生成、回測、診斷的完整結果
        """
        # Step 1: 解析意圖
        intent = self._parse_intent(nl_request)

        # Step 2: 股票篩選
        matched_stocks = self._screen_stocks(intent)

        # Step 3: 策略生成 + 回測 (對每個股票)
        strategy_results: list[StrategyResult] = []
        for stock in matched_stocks[:3]:  # 最多 3 支股票
            nl_with_stock = f"{nl_request} — 分析 {stock.name} ({stock.symbol})"
            sr = self._generate_and_backtest(nl_with_stock, stock)
            strategy_results.append(sr)

        # Step 4: 診斷
        diagnostic_results = [self._diagnose(sr) for sr in strategy_results]

        # Step 5: 生成摘要
        summary = self._generate_summary(nl_request, matched_stocks, strategy_results, diagnostic_results)

        return PipelineResult(
            original_request=nl_request,
            parsed_intent=intent,
            matched_stocks=matched_stocks,
            strategy_results=strategy_results,
            diagnostic_results=diagnostic_results,
            summary=summary,
        )

    def _generate_summary(
        self,
        nl_request: str,
        stocks: list[StockResult],
        strategies: list[StrategyResult],
        diagnostics: list[DiagnosticResult],
    ) -> str:
        """生成 AI 摘要"""
        if not diagnostics:
            return "No results to summarize."

        avg_score = sum(d.score for d in diagnostics) / len(diagnostics)
        best_score = max(d.score for d in diagnostics)
        best_stock = stocks[diagnostics.index(max(diagnostics, key=lambda d: d.score))] if stocks else None

        summary_parts = [
            f"请求: {nl_request}",
            f"找到 {len(stocks)} 支符合条件的股票",
            f"生成了 {len(strategies)} 个策略",
            f"平均诊后评分: {avg_score:.0f}/100",
            f"最佳股票: {best_stock.name if best_stock else 'N/A'} (评分: {best_score}/100)",
        ]

        return "\n".join(summary_parts)

    @staticmethod
    def _get_demo_symbols(scope: str, limit: int) -> list[str]:
        """返回演示用股票代碼"""
        demo_map = {
            "A股": ["000001", "600036", "601318", "601166", "002142"],
            "美股": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        }
        base = demo_map.get(scope, demo_map["A股"])
        return base[:limit]