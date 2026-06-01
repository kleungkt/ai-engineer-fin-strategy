"""
Strategy Agent module for AI Strategy Generator.

AI-powered agent that parses natural language strategy descriptions,
generates Backtrader code, validates, and backtests them.
"""

import json
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from backtester import BacktestResult, run_backtest
from code_validator import validate_backtrader_strategy, extract_class_name


class StrategySpec(BaseModel):
    """Structured specification of a trading strategy."""

    name: str = Field(description="Strategy name")
    description: str = Field(description="Human-readable description of the strategy")
    strategy_type: str = Field(description="Type: trend_following, mean_reversion, breakout, momentum")
    entry_rules: list[str] = Field(description="List of entry conditions")
    exit_rules: list[str] = Field(description="List of exit conditions")
    risk_management: dict[str, Any] = Field(
        default_factory=dict,
        description="Risk management rules (stop_loss, take_profit, position_size, etc.)",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters (periods, thresholds, etc.)",
    )


# OpenAI function schema for StrategySpec
STRATEGY_SPEC_FUNCTION = {
    "name": "define_strategy",
    "description": "Define a structured trading strategy specification from a natural language description.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name for the strategy",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the strategy logic",
            },
            "strategy_type": {
                "type": "string",
                "enum": ["trend_following", "mean_reversion", "breakout", "momentum"],
                "description": "Category of the strategy",
            },
            "entry_rules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of conditions to enter a position",
            },
            "exit_rules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of conditions to exit a position",
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "stop_loss": {"type": "number", "description": "Stop loss percentage (e.g., 0.05 for 5%)"},
                    "take_profit": {"type": "number", "description": "Take profit percentage"},
                    "position_size": {"type": "number", "description": "Fraction of portfolio to allocate (0-1)"},
                },
            },
            "params": {
                "type": "object",
                "description": "Strategy-specific parameters (indicator periods, thresholds, etc.)",
                "additionalProperties": True,
            },
        },
        "required": ["name", "description", "strategy_type", "entry_rules", "exit_rules"],
    },
}

# System prompt for code generation
CODE_GEN_SYSTEM = """You are an expert quantitative developer specializing in the Backtrader framework.
Given a strategy specification, generate complete, runnable Backtrader Strategy Python code.

Requirements:
1. Import backtrader as bt
2. Define a single class that inherits from bt.Strategy
3. Define params tuple with all configurable parameters
4. Implement __init__ with indicator setup
5. Implement next with trading logic
6. Use self.buy(), self.sell(), self.close() for order execution
7. Check self.position for current holdings
8. Use self.data.close[0] for current close price
9. Use self.broker.getcash() for available cash
10. Include docstrings

Return ONLY the Python code, no explanations or markdown fences.
"""


class StrategyAgent:
    """
    AI agent for parsing, generating, validating, and executing trading strategies.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        """
        Initialize the StrategyAgent.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def parse_strategy(self, user_input: str) -> StrategySpec:
        """
        Parse a natural language strategy description into a structured StrategySpec.

        Uses OpenAI function calling to extract structured data.

        Args:
            user_input: Natural language description of a trading strategy

        Returns:
            StrategySpec with parsed strategy details
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a trading strategy analyst. Parse user descriptions into structured strategy specifications.",
                },
                {"role": "user", "content": user_input},
            ],
            tools=[{"type": "function", "function": STRATEGY_SPEC_FUNCTION}],
            tool_choice={"type": "function", "function": {"name": "define_strategy"}},
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("LLM did not return a function call")

        args = json.loads(message.tool_calls[0].function.arguments)
        return StrategySpec(**args)

    def generate_code(self, spec: StrategySpec) -> str:
        """
        Generate Backtrader strategy code from a StrategySpec.

        Args:
            spec: Structured strategy specification

        Returns:
            Python code string defining a Backtrader Strategy class
        """
        spec_text = f"""
Strategy Name: {spec.name}
Type: {spec.strategy_type}
Description: {spec.description}
Entry Rules:
{chr(10).join(f'- {r}' for r in spec.entry_rules)}
Exit Rules:
{chr(10).join(f'- {r}' for r in spec.exit_rules)}
Risk Management: {json.dumps(spec.risk_management)}
Parameters: {json.dumps(spec.params)}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODE_GEN_SYSTEM},
                {"role": "user", "content": spec_text},
            ],
            temperature=0.2,
        )

        code = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate generated strategy code.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, message)
        """
        return validate_backtrader_strategy(code)

    def execute_strategy(
        self, code: str, data: pd.DataFrame, initial_cash: float = 100000
    ) -> BacktestResult:
        """
        Execute a strategy against historical data via backtesting.

        Args:
            code: Validated Backtrader strategy code
            data: DataFrame with OHLCV data
            initial_cash: Starting cash amount

        Returns:
            BacktestResult with performance metrics
        """
        return run_backtest(data=data, strategy_code=code, initial_cash=initial_cash)

    def run_pipeline(
        self, user_input: str, data: pd.DataFrame, initial_cash: float = 100000
    ) -> dict[str, Any]:
        """
        Run the full strategy generation and testing pipeline.

        Steps:
        1. Parse natural language input into StrategySpec
        2. Generate Backtrader code from the spec
        3. Validate the generated code
        4. Backtest against the provided data
        5. Return all artifacts

        Args:
            user_input: Natural language strategy description
            data: DataFrame with OHLCV data
            initial_cash: Starting cash amount

        Returns:
            Dict with keys: spec, code, is_valid, validation_msg, result
        """
        # Step 1: Parse
        spec = self.parse_strategy(user_input)

        # Step 2: Generate code
        code = self.generate_code(spec)

        # Step 3: Validate
        is_valid, msg = self.validate_code(code)

        # Step 4: Backtest (if valid)
        result = None
        if is_valid:
            result = self.execute_strategy(code, data, initial_cash)

        return {
            "spec": spec,
            "code": code,
            "is_valid": is_valid,
            "validation_msg": msg,
            "result": result,
        }
