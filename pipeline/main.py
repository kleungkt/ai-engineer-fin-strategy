#!/usr/bin/env python3
"""
Unified Pipeline CLI
=====================
Command-line interface for the E2E pipeline.

Usage:
    python pipeline/main.py analyze "帮我找最近 RSI 低於 30 的 A 股"
    python pipeline/main.py analyze "MACD 金叉且 RSI < 30 的股票，回測均值回歸策略" --live
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add pipeline dir to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_pipeline import UnifiedPipeline


def analyze(nl_request: str, demo: bool = True, max_stocks: int = 10, llm_model: str = "gpt-4o-mini"):
    """執行分析並打印結果"""
    print(f"\n🎯 請求: {nl_request}")
    print(f"{'─' * 60}")
    print(f"模式: {'演示' if demo else '真實 API'}")
    print()

    pipeline = UnifiedPipeline(
        llm_model=llm_model,
        max_stocks=max_stocks,
        demo_mode=demo,
    )

    start = time.perf_counter()
    result = pipeline.run(nl_request)
    elapsed = (time.perf_counter() - start) * 1000

    # ── Intent ─────────────────────────────────────────────
    print(f"📌 意圖解析 ({elapsed:.0f}ms)")
    print(f"   類型: {result.parsed_intent.get('intent_type', 'N/A')}")
    indicators = result.parsed_intent.get("indicators", [])
    if indicators:
        print(f"   指標: {', '.join(str(i) for i in indicators)}")

    # ── Stocks ─────────────────────────────────────────────
    print(f"\n📊 股票篩選結果 ({len(result.matched_stocks)} 支)")
    if not result.matched_stocks:
        print("   未找到符合條件的股票")
    else:
        for i, stock in enumerate(result.matched_stocks, 1):
            print(f"   {i}. {stock.name} ({stock.symbol})")
            for cond in stock.matched_conditions:
                print(f"      • {cond}")
            if stock.metrics:
                for k, v in stock.metrics.items():
                    print(f"      • {k}: {v:.2f}")

    # ── Strategy + Backtest ─────────────────────────────────
    print(f"\n🤖 策略生成 + 回測 ({len(result.strategy_results)} 個策略)")
    for i, sr in enumerate(result.strategy_results, 1):
        status = "✅" if sr.is_valid else "❌"
        print(f"\n   {i}. {status} {sr.natural_language[:60]}")
        if sr.backtest_result:
            bt = sr.backtest_result
            ret = bt.get("total_return", 0)
            color = "🟢" if ret >= 0 else "🔴"
            print(f"      {color} 總收益: {ret*100:+.2f}%")
            print(f"      📈 夏普: {bt.get('sharpe_ratio', 0):.2f}")
            print(f"      📉 最大回撤: {bt.get('max_drawdown', 0)*100:.2f}%")
            print(f"      🎯 勝率: {bt.get('win_rate', 0)*100:.1f}%")
            print(f"      📋 交易次數: {bt.get('total_trades', 0)}")
        elif sr.error_message:
            print(f"      ❌ 錯誤: {sr.error_message}")

    # ── Diagnostics ─────────────────────────────────────────
    print(f"\n🏥 策略診斷 ({len(result.diagnostic_results)} 份報告)")
    for i, diag in enumerate(result.diagnostic_results, 1):
        emoji = "🟢" if diag.score >= 70 else "🟡" if diag.score >= 50 else "🔴"
        print(f"   {i}. {emoji} 評分: {diag.score}/100 ({diag.rating})")
        if diag.metrics_rating:
            for metric, rating in diag.metrics_rating.items():
                icon = "✅" if rating in ("good", "excellent") else "⚠️" if rating == "acceptable" else "❌"
                print(f"      {icon} {metric}: {rating}")
        if diag.ai_analysis:
            print(f"      💡 {diag.ai_analysis[:150]}...")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"📝 摘要:")
    for line in result.summary.split("\n"):
        print(f"   {line}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AI Strategy Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    cmd = sub.add_parser("analyze", help="執行 E2E 分析")
    cmd.add_argument("nl_request", help="自然語言請求")
    cmd.add_argument("--live", action="store_true", help="使用真實 API（非演示模式）")
    cmd.add_argument("--max-stocks", type=int, default=10, help="最大股票數")
    cmd.add_argument("--model", default="gpt-4o-mini", help="LLM 模型")
    cmd.add_argument("--json", action="store_true", help="JSON 輸出")

    args = parser.parse_args()

    if args.command == "analyze":
        exit_code = analyze(
            args.nl_request,
            demo=not args.live,
            max_stocks=args.max_stocks,
            llm_model=args.model,
        )
        sys.exit(exit_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()