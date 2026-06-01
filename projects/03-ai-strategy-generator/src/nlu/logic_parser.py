"""
LLM-based trading logic parser for strategy descriptions.

Parses natural language trading strategy descriptions into structured
``TradingLogic`` objects with entry/exit conditions, position sizing rules,
and risk management rules. Uses OpenAI function calling and supports both
Chinese and English inputs.

Typical usage::

    logic = parse_logic("當RSI低於30且成交量放大時買入，跌破20日均線賣出，設置2%止損")
    print(logic.entry_conditions[0].left_operand)   # "RSI"
    print(logic.entry_conditions[0].operator)        # "<"
    print(logic.entry_conditions[0].right_operand)   # "30"
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ComparisonOperator(str, Enum):
    """Supported comparison operators for trading conditions."""

    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSSOVER = "crossover"
    CROSSUNDER = "crossunder"


class Conjunction(str, Enum):
    """Logical conjunctions for combining conditions."""

    AND = "AND"
    OR = "OR"


class PositionSizing(str, Enum):
    """Supported position sizing methods."""

    FIXED = "fixed"          # 固定金額/股數
    PERCENT = "percent"      # 資金百分比
    KELLY = "kelly"          # 凱利公式


class Condition(BaseModel):
    """A single trading condition.

    Attributes:
        left_operand: Left side of the condition, typically an indicator
            or expression, e.g. ``"RSI"``, ``"MA(20)"``, ``"close"``.
        operator: Comparison operator.
        right_operand: Right side of the condition, typically a value or
            indicator, e.g. ``"30"``, ``"MA(60)"``, ``"upper_band"``.
        conjunction: Logical connector to the next condition (``None`` for
            the last condition in a group).
    """

    left_operand: str = Field(
        ...,
        description=(
            "條件左側，通常是技術指標或表達式，"
            "例如 'RSI', 'MA(20)', 'close', 'VOLUME'"
        ),
    )
    operator: ComparisonOperator = Field(
        ...,
        description=(
            "比較運算子: >（大於）, <（小於）, >=（大於等於）, <=（小於等於）, "
            "==（等於）, crossover（向上穿越/黃金交叉）, crossunder（向下穿越/死亡交叉）"
        ),
    )
    right_operand: str = Field(
        ...,
        description=(
            "條件右側，通常是數值或另一個指標，"
            "例如 '30', 'MA(60)', 'upper_band', '0.02'"
        ),
    )
    conjunction: Optional[Conjunction] = Field(
        default=None,
        description=(
            "與下一個條件的邏輯連接詞: AND（且）, OR（或）。"
            "最後一個條件設為 null。"
        ),
    )

    @field_validator("left_operand", "right_operand")
    @classmethod
    def validate_operand(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("operand must be a non-empty string")
        return v.strip()


class TradingLogic(BaseModel):
    """Structured trading logic parsed from natural language.

    Attributes:
        entry_conditions: List of conditions that must be met to enter
            a position (buy/long).
        exit_conditions: List of conditions that trigger exiting a
            position (sell/short).
        position_sizing: Method for determining position size.
        risk_rules: List of risk management conditions (stop loss,
            take profit, trailing stop, etc.).
    """

    entry_conditions: List[Condition] = Field(
        default_factory=list,
        description="進場條件列表（買入/做多觸發條件）",
    )
    exit_conditions: List[Condition] = Field(
        default_factory=list,
        description="出場條件列表（賣出/平倉觸發條件）",
    )
    position_sizing: PositionSizing = Field(
        default=PositionSizing.PERCENT,
        description=(
            "倉位管理方式: fixed（固定金額）, percent（資金百分比）, "
            "kelly（凱利公式）"
        ),
    )
    risk_rules: List[Condition] = Field(
        default_factory=list,
        description=(
            "風險管理規則列表，例如止損、止盈、移動止損等。"
            "這些條件通常在進場後自動生效。"
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

_FUNCTION_SCHEMA: Dict[str, Any] = {
    "name": "parse_trading_logic",
    "description": (
        "將自然語言交易策略描述解析為結構化的交易邏輯，"
        "包括進場條件、出場條件、倉位管理和風險規則。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entry_conditions": {
                "type": "array",
                "description": "進場（買入/做多）條件列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "left_operand": {
                            "type": "string",
                            "description": (
                                "條件左側，指標或表達式，"
                                "如 'RSI', 'MA(20)', 'close', 'VOLUME'"
                            ),
                        },
                        "operator": {
                            "type": "string",
                            "enum": [
                                ">", "<", ">=", "<=", "==",
                                "crossover", "crossunder",
                            ],
                            "description": (
                                "比較運算子: >, <, >=, <=, ==, "
                                "crossover（向上穿越）, crossunder（向下穿越）"
                            ),
                        },
                        "right_operand": {
                            "type": "string",
                            "description": "條件右側，數值或指標",
                        },
                        "conjunction": {
                            "type": "string",
                            "enum": ["AND", "OR"],
                            "description": (
                                "與下一個條件的邏輯連接: AND（且）, OR（或）。"
                                "最後一個條件填 null。"
                            ),
                        },
                    },
                    "required": [
                        "left_operand",
                        "operator",
                        "right_operand",
                    ],
                },
            },
            "exit_conditions": {
                "type": "array",
                "description": "出場（賣出/平倉）條件列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "left_operand": {
                            "type": "string",
                            "description": "條件左側，指標或表達式",
                        },
                        "operator": {
                            "type": "string",
                            "enum": [
                                ">", "<", ">=", "<=", "==",
                                "crossover", "crossunder",
                            ],
                            "description": "比較運算子",
                        },
                        "right_operand": {
                            "type": "string",
                            "description": "條件右側，數值或指標",
                        },
                        "conjunction": {
                            "type": "string",
                            "enum": ["AND", "OR"],
                            "description": "邏輯連接詞，最後一個條件填 null",
                        },
                    },
                    "required": [
                        "left_operand",
                        "operator",
                        "right_operand",
                    ],
                },
            },
            "position_sizing": {
                "type": "string",
                "enum": ["fixed", "percent", "kelly"],
                "description": (
                    "倉位管理方式: "
                    "fixed（固定金額/股數）, "
                    "percent（資金百分比，如投入30%資金）, "
                    "kelly（凱利公式計算最優倉位）"
                ),
                "default": "percent",
            },
            "risk_rules": {
                "type": "array",
                "description": (
                    "風險管理規則。這些規則在進場後自動生效，"
                    "例如止損、止盈、移動止損等。"
                    "用 left_operand 表示風險類型，operator 和 right_operand 表示觸發條件。"
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "left_operand": {
                            "type": "string",
                            "description": (
                                "風險類型或計算基準，"
                                "如 'stop_loss', 'take_profit', 'trailing_stop', "
                                "'entry_price'"
                            ),
                        },
                        "operator": {
                            "type": "string",
                            "enum": [
                                ">", "<", ">=", "<=", "==",
                                "crossover", "crossunder",
                            ],
                            "description": "比較運算子",
                        },
                        "right_operand": {
                            "type": "string",
                            "description": (
                                "觸發值，如 '0.02'（2%止損）, '0.05'（5%止盈）, "
                                "'MA(20)'（跌破均線止損）"
                            ),
                        },
                        "conjunction": {
                            "type": "string",
                            "enum": ["AND", "OR"],
                            "description": "邏輯連接詞，最後一個條件填 null",
                        },
                    },
                    "required": [
                        "left_operand",
                        "operator",
                        "right_operand",
                    ],
                },
            },
        },
        "required": [
            "entry_conditions",
            "exit_conditions",
            "position_sizing",
            "risk_rules",
        ],
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一個專業的量化交易邏輯解析器，專門將自然語言策略描述轉換為結構化的交易邏輯。
你支援中文和英文輸入。

## 解析規則

### 1. 進場條件 (entry_conditions)
識別買入/做多/建倉的觸發條件。常見表達:
- 「當...時買入」「...的時候進場」
- "buy when...", "enter long when..."
- 「低於...時」「突破...時」
- 分批建倉要拆分為多個條件

條件格式:
- left_operand: 指標名稱或表達式（如 RSI, MA(20), close, VOLUME, BOLL.lower）
- operator: 比較運算子（>, <, >=, <=, ==, crossover, crossunder）
- right_operand: 數值或另一個指標（如 30, MA(60), upper_band）
- conjunction: AND/OR 連接多個條件（最後一個條件為 null）

### 2. 出場條件 (exit_conditions)
識別賣出/平倉/減倉的觸發條件。常見表達:
- 「跌破...時賣出」「觸及...時出場」
- "sell when...", "exit when..."
- 「高於...時減倉」

### 3. 倉位管理 (position_sizing)
- fixed: 固定金額或股數（「每次買100股」）
- percent: 資金百分比（「投入30%資金」，預設）
- kelly: 凱利公式（「用凱利公式計算倉位」）

### 4. 風險管理規則 (risk_rules)
識別止損、止盈、移動止損等風險控制規則。常見表達:
- 「設置N%止損」 → left_operand="stop_loss", operator="<", right_operand="entry_price * 0.97"
- 「止盈N%」 → left_operand="take_profit", operator=">", right_operand="entry_price * 1.05"
- 「跌破均線止損」 → left_operand="close", operator="<", right_operand="MA(20)", conjunction=null
- 「移動止損N%」 → left_operand="trailing_stop", operator="<", right_operand="0.02"

注意: 止損/止盈的 right_operand 應盡量表達為相對於進場價格的公式或百分比。

## 常見指標表達式
- RSI, RSI(14) - 相對強弱指標
- MA(20), MA(60), EMA(20) - 移動平均線
- MACD, MACD.signal - MACD及其信號線
- BOLL, BOLL.upper, BOLL.lower - 布林通道
- KD.K, KD.D - 隨機指標
- VOLUME - 成交量
- ATR(14) - 平均真實範圍
- close, open, high, low - 價格

## 中文特殊處理
- 「低於」「小於」 → operator: <
- 「高於」「大於」 → operator: >
- 「突破」「上穿」「黃金交叉」 → operator: crossover
- 「跌破」「下穿」「死亡交叉」 → operator: crossunder
- 「且」「同時」 → conjunction: AND
- 「或」 → conjunction: OR
- 「分批建倉」 → 多個 entry_conditions
- 「止損N%」 → risk_rules 中的 stop_loss 條件

## 輸出要求
- 只輸出 function call，不要自行回答問題
- 每個條件的 conjunction 指向下一個條件的邏輯關係，最後一個為 null
- 如果文本中沒有明確提到倉位管理，預設為 percent
- 風險規則中的止損止盈要盡量轉換為可計算的表達式
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logic_prompt() -> str:
    """Return the system prompt used for trading logic parsing.

    Useful for debugging, logging, or displaying the prompt in a UI.

    Returns:
        The full system prompt string.
    """
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------


def parse_logic(
    text: str,
    *,
    model: str = "gpt-4o-mini",
) -> TradingLogic:
    """Parse a natural-language trading strategy into structured logic.

    Args:
        text: Free-form text describing a trading strategy, e.g.
            ``"當RSI低於30且成交量放大時買入，跌破20日均線賣出，設置2%止損"``.
        model: OpenAI model to use (default ``"gpt-4o-mini"``).

    Returns:
        A validated :class:`TradingLogic` instance.

    Raises:
        ValueError: If the input is empty or the LLM response cannot be
            parsed or validated.
        RuntimeError: If the OpenAI API call fails.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    client = OpenAI()  # 透過環境變數 OPENAI_API_KEY 讀取金鑰

    # 呼叫 OpenAI function calling
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=[{"type": "function", "function": _FUNCTION_SCHEMA}],
            tool_choice={
                "type": "function",
                "function": {"name": "parse_trading_logic"},
            },
            temperature=0.0,  # 確保穩定輸出
        )
    except Exception as exc:
        logger.error("OpenAI API 呼叫失敗: %s", exc)
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    # 解析 function call 回應
    choice = response.choices[0]
    message = choice.message

    # 確認 LLM 有回傳 tool_calls
    if not message.tool_calls:
        raise ValueError(
            "LLM did not return a tool call. "
            f"Finish reason: {choice.finish_reason}. "
            f"Content: {message.content!r}"
        )

    tool_call = message.tool_calls[0]
    try:
        raw_args: Dict[str, Any] = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse LLM function arguments as JSON: {exc}"
        ) from exc

    logger.debug(
        "LLM 邏輯解析結果: %s", json.dumps(raw_args, ensure_ascii=False)
    )

    # 使用 Pydantic 驗證並建構 TradingLogic
    try:
        logic = TradingLogic.model_validate(raw_args)
    except Exception as exc:
        raise ValueError(
            f"LLM output failed Pydantic validation: {exc}"
        ) from exc

    return logic


# ---------------------------------------------------------------------------
# Convenience: module-level demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 快速測試（需設定 OPENAI_API_KEY 環境變數）
    sample_strategies = [
        "當RSI低於30且成交量放大時買入，跌破20日均線賣出，設置2%止損",
        "布林通道下軌且RSI超賣時分批建倉",
        "MACD黃金交叉買入，死亡交叉賣出，止損3%，止盈10%",
        "Buy when EMA20 crosses above EMA50, sell when price drops below MA(20), 5% stop loss",
        "用凱利公式計算倉位，RSI超買時減倉一半",
    ]
    for desc in sample_strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {desc}")
        try:
            result = parse_logic(desc)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
