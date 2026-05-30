#!/usr/bin/env python3
"""
Natural Language Stock Screener — Interactive CLI Entry Point.

Usage:
    python main.py          # interactive REPL
    python main.py --demo   # run preset demo queries
"""

import argparse

from parser import parse_query
from screener import generate_sample_data, screen_stocks, ScreenResult

# ---------------------------------------------------------------------------
# Demo stock universe
# ---------------------------------------------------------------------------
DEMO_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "V", "MA",
    "JNJ", "PFE", "UNH", "KO", "PEP",
]


def _build_sample_data(symbols: list[str] | None = None, days: int = 120) -> dict:
    """Generate sample OHLCV data for every symbol."""
    symbols = symbols or DEMO_STOCKS
    return {sym: generate_sample_data(sym, days=days) for sym in symbols}


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _print_results(results: list[ScreenResult], verbose: bool = False) -> None:
    """Pretty-print screening results to stdout."""
    matched = [r for r in results if r.matched]
    failed = [r for r in results if not r.matched]

    print("\n" + "=" * 60)
    print(f"  RESULTS: {len(matched)} matched / {len(results)} total")
    print("=" * 60)

    if matched:
        print("\n✅  MATCHED STOCKS:\n")
        for r in matched:
            print(f"  ★  {r.symbol}")
            if r.explanation:
                for part in r.explanation.split(";"):
                    print(f"       {part.strip()}")
            if verbose:
                print(f"       Indicators: {r.indicators}")
            print()
    else:
        print("\n  (no stocks matched)\n")

    if verbose and failed:
        print("❌  DID NOT MATCH:\n")
        for r in failed:
            print(f"  ·  {r.symbol}")
            if r.explanation:
                for part in r.explanation.split(";"):
                    print(f"       {part.strip()}")
            print()

    print("-" * 60)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive(symbols: list[str] | None = None, days: int = 120) -> None:
    """
    Run the interactive stock-screener loop.

    Type a natural-language query and see which stocks pass.
    Type 'quit' or 'exit' to leave.
    """
    print("\n📈  Natural Language Stock Screener")
    print("=" * 50)
    print("Describe the stocks you're looking for in plain language.")
    print("Examples:")
    print("  · \"RSI below 30\"")
    print("  · \"MACD golden cross\"")
    print("  · \"SMA 20 above\"")
    print("Type 'quit' to exit.\n")

    stock_data = _build_sample_data(symbols, days)

    while True:
        try:
            user_input = input("🔍  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Parse natural language into structured intent
        try:
            query = parse_query(user_input)
        except Exception as exc:
            print(f"  ⚠  Could not parse query: {exc}\n")
            continue

        # Show parsed summary
        first = query.indicators[0] if query.indicators else None
        if first:
            print(
                f"\n  Parsed → indicator={first.name}, "
                f"comparison={first.comparison}, "
                f"value={first.value}, "
                f"params={first.params}"
            )

        # Screen
        results = screen_stocks(query, stock_data)
        _print_results(results)


# ---------------------------------------------------------------------------
# Preset demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Run a handful of preset queries and display results."""
    stock_data = _build_sample_data()

    preset_queries = [
        "RSI below 30",
        "MACD golden cross",
        "SMA 20 above",
        "price above 100",
    ]

    for q in preset_queries:
        print(f"\n{'#' * 60}")
        print(f"  DEMO QUERY: \"{q}\"")
        print(f"{'#' * 60}")

        try:
            query = parse_query(q)
        except Exception as exc:
            print(f"  ⚠  Parse error: {exc}")
            continue

        results = screen_stocks(query, stock_data)
        _print_results(results, verbose=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Natural Language Stock Screener")
    ap.add_argument("--demo", action="store_true", help="Run preset demo queries")
    ap.add_argument("--days", type=int, default=120, help="Days of sample data (default 120)")
    args = ap.parse_args()

    if args.demo:
        demo()
    else:
        interactive(days=args.days)


if __name__ == "__main__":
    main()
