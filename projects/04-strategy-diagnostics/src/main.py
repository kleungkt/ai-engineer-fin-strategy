"""CLI entry point for strategy diagnostics."""

import argparse
import json
import sys

from .evaluator import evaluate_backtest
from .formatter import format_report
from .models import BacktestResult


def main() -> None:
    """Run diagnostics from CLI."""
    parser = argparse.ArgumentParser(description="Strategy Diagnostics Tool")
    parser.add_argument("file", help="Path to JSON file with backtest results")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Generate AI analysis (requires OpenAI API key)",
    )
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle single result or list of results
    if isinstance(data, list):
        results = [BacktestResult(**item) for item in data]
    else:
        results = [BacktestResult(**data)]

    for i, result in enumerate(results):
        if len(results) > 1:
            print(f"\n{'#' * 60}")
            print(f"  Strategy {i + 1}")
            print(f"{'#' * 60}")

        report = evaluate_backtest(result)

        if args.ai:
            try:
                from .ai_analyst import generate_analysis

                analysis = generate_analysis(report)
                report.ai_analysis = analysis
            except Exception as e:
                print(f"Warning: AI analysis failed: {e}", file=sys.stderr)

        output = format_report(report, format=args.format)
        print(output)


if __name__ == "__main__":
    main()
