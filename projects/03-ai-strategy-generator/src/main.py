"""
Interactive CLI for AI Strategy Generator.

Supports strategy generation from natural language, template execution,
and parameter optimization.
"""

import argparse
import json
import sys

import pandas as pd

from backtester import run_backtest
from data_fetcher import fetch_stock_daily, generate_sample_data
from optimizer import grid_search
from strategy_agent import StrategyAgent
from strategy_templates import list_templates, render_template


def print_result(result, label: str = "Backtest Result") -> None:
    """Pretty-print a BacktestResult."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total Return:    {result.total_return*100:.2f}%")
    print(f"  Annual Return:   {result.annual_return*100:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:    {result.max_drawdown*100:.2f}%")
    print(f"  Win Rate:        {result.win_rate*100:.1f}%")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"{'='*60}\n")


def print_optimization_result(opt_result) -> None:
    """Pretty-print an OptimizationResult."""
    print(f"\n{'='*60}")
    print("  Optimization Results")
    print(f"{'='*60}")
    print(f"  Metric:            {opt_result.metric}")
    print(f"  Best Score:        {opt_result.best_score:.6f}")
    print(f"  Best Params:       {json.dumps(opt_result.best_params, indent=2)}")
    print(f"  Combinations:      {opt_result.total_combinations}")
    print(f"{'='*60}")

    if opt_result.best_backtest:
        print("\n  Best Run Performance:")
        print_result(opt_result.best_backtest, "Best Parameter Set")

    if opt_result.all_results:
        print("  Top 5 Results:")
        for i, r in enumerate(opt_result.all_results[:5], 1):
            print(f"    {i}. score={r.get('score', 'N/A')}, params={r.get('params', {})}")
    print()


def get_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch data with fallback to sample data."""
    try:
        data = fetch_stock_daily(symbol, days)
        print(f"  Fetched {len(data)} bars for {symbol}")
        return data
    except Exception as e:
        print(f"  Could not fetch live data ({e}), using sample data")
        data = generate_sample_data(symbol, days)
        print(f"  Generated {len(data)} sample bars")
        return data


def cmd_generate(args) -> None:
    """Generate a strategy from natural language."""
    print(f"\n  Generating strategy from: '{args.description}'")
    print(f"  Symbol: {args.symbol}, Days: {args.days}, Cash: {args.initial_cash}")

    data = get_data(args.symbol, args.days)

    agent = StrategyAgent(model=args.model)
    result = agent.run_pipeline(args.description, data, args.initial_cash)

    print(f"\n  Strategy Spec:")
    print(f"    Name:        {result['spec'].name}")
    print(f"    Type:        {result['spec'].strategy_type}")
    print(f"    Description: {result['spec'].description}")
    print(f"    Entry Rules: {result['spec'].entry_rules}")
    print(f"    Exit Rules:  {result['spec'].exit_rules}")

    print(f"\n  Validation: {'PASS' if result['is_valid'] else 'FAIL'} - {result['validation_msg']}")

    if result["result"]:
        print_result(result["result"], f"Backtest: {result['spec'].name}")

    if args.save_code:
        with open(args.save_code, "w") as f:
            f.write(result["code"])
        print(f"  Code saved to: {args.save_code}")


def cmd_template(args) -> None:
    """Run a strategy template."""
    if args.list:
        print("\n  Available Templates:")
        for t in list_templates():
            print(f"    - {t['name']}: {t['description']}")
        return

    if not args.template:
        print("  Error: --template is required (or use --list)")
        return

    print(f"\n  Running template: {args.template}")

    # Parse custom params
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            print("  Error: --params must be valid JSON")
            return

    code = render_template(args.template, params)
    data = get_data(args.symbol, args.days)

    result = run_backtest(data, code, initial_cash=args.initial_cash)
    print_result(result, f"Template: {args.template}")

    if args.save_code:
        with open(args.save_code, "w") as f:
            f.write(code)
        print(f"  Code saved to: {args.save_code}")


def cmd_optimize(args) -> None:
    """Optimize strategy parameters."""
    if not args.template and not args.code_file:
        print("  Error: provide --template or --code-file")
        return

    print(f"\n  Optimizing strategy...")
    print(f"  Metric: {args.metric}, Cash: {args.initial_cash}")

    # Get strategy code
    if args.template:
        code = render_template(args.template)
        strategy_type = args.template
    else:
        with open(args.code_file, "r") as f:
            code = f.read()
        strategy_type = args.strategy_type

    # Get param grid
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    elif strategy_type:
        param_grid = suggest_params(strategy_type)
        print(f"  Using suggested params for '{strategy_type}': {list(param_grid.keys())}")
    else:
        print("  Error: provide --param-grid or --strategy-type")
        return

    data = get_data(args.symbol, args.days)

    opt_result = grid_search(
        strategy_code=code,
        data=data,
        param_grid=param_grid,
        metric=args.metric,
        initial_cash=args.initial_cash,
    )

    print_optimization_result(opt_result)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="AI Strategy Generator - Generate, test, and optimize trading strategies",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="000001", help="Stock symbol (default: 000001)")
    common.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    common.add_argument("--initial-cash", type=float, default=100000, help="Starting cash (default: 100000)")
    common.add_argument("--save-code", help="Save generated/rendered code to file")

    # Generate command
    gen_parser = subparsers.add_parser("generate", parents=[common], help="Generate strategy from description")
    gen_parser.add_argument("description", help="Natural language strategy description")
    gen_parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")

    # Template command
    tpl_parser = subparsers.add_parser("template", parents=[common], help="Run a strategy template")
    tpl_parser.add_argument("--template", "-t", help="Template name")
    tpl_parser.add_argument("--list", "-l", action="store_true", help="List available templates")
    tpl_parser.add_argument("--params", "-p", help="Template parameters as JSON string")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", parents=[common], help="Optimize strategy parameters")
    opt_parser.add_argument("--template", "-t", help="Template name to optimize")
    opt_parser.add_argument("--code-file", help="Path to strategy code file")
    opt_parser.add_argument("--strategy-type", help="Strategy type for suggested params")
    opt_parser.add_argument("--param-grid", help="Custom param grid as JSON string")
    opt_parser.add_argument("--metric", default="sharpe_ratio",
                            choices=["sharpe_ratio", "total_return", "annual_return", "max_drawdown", "win_rate"],
                            help="Metric to optimize (default: sharpe_ratio)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "template":
        cmd_template(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
