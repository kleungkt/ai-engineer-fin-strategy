"""
LLM-based parser for natural language stock queries.

This module uses OpenAI function calling (gpt-4o-mini) to parse free-form
Chinese/English stock screening questions into structured `QueryIntent`
objects. The structured intent can then be consumed by downstream screener
logic to filter and match stocks.

Typical usage:

    intent = parse_query("找出 MACD 黃金交叉且 RSI < 30 的 A 股，近 30 天")
    print(intent.intent_type)       # "indicator_screener"
    print(intent.indicators[0])     # IndicatorCondition(name="MACD", ...)
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
# Pydantic models – structured output definitions
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    """Supported query intent types."""
    INDICATOR_SCREENER = "indicator_screener"
    PATTERN_QUERY = "pattern_query"
    FUNDAMENTAL_QUERY = "fundamental_query"


class IndicatorCondition(BaseModel):
    """A single technical-indicator condition extracted from user input.

    Attributes:
        name: Indicator name, e.g. ``"MACD"``, ``"RSI"``, ``"MA"``, ``"KD"``.
        comparison: Comparison direction – ``"above"``, ``"below"``,
            ``"crossover"``, or ``"crossunder"``.
        value: Threshold value (optional).  May be omitted for cross-type
            conditions where only signal-line / parameter-based detection is
            needed.
        params: Indicator hyper-parameters such as fast/slow/signal periods.
            Keys and values are flexible (e.g. ``{"fast": 12, "slow": 26}``).
    """

    name: str = Field(
        ...,
        description="技術指標名稱，如 MACD, RSI, MA, KD, BOLL, ATR, VWAP 等",
    )
    comparison: str = Field(
        ...,
        description="比較方向: above / below / crossover / crossunder",
    )
    value: Optional[float] = Field(
        default=None,
        description="閾值數值（可選），例如 RSI < 30 的 30",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="指標參數，例如 {fast: 12, slow: 26, signal: 9}",
    )

    @field_validator("comparison")
    @classmethod
    def validate_comparison(cls, v: str) -> str:
        allowed = {"above", "below", "crossover", "crossunder"}
        if v.lower() not in allowed:
            raise ValueError(
                f"comparison must be one of {allowed}, got '{v}'"
            )
        return v.lower()


class QueryIntent(BaseModel):
    """Parsed intent from a natural-language stock query.

    Attributes:
        intent_type: Category of the user's question.
        indicators: List of technical indicator conditions to evaluate.
        stock_scope: Market filter, e.g. ``"A股"``, ``"美股"``, ``"加密貨幣"``.
        time_range: Look-back window in days (e.g. 30 means the last 30
            trading days).
    """

    intent_type: IntentType = Field(
        ...,
        description=(
            "查詢類型: indicator_screener（技術面選股）, "
            "pattern_query（K線型態查詢）, fundamental_query（基本面查詢）"
        ),
    )
    indicators: List[IndicatorCondition] = Field(
        default_factory=list,
        description="技術指標條件列表",
    )
    stock_scope: str = Field(
        default="A股",
        description="股票市場範圍，如 A股、美股、加密貨幣",
    )
    time_range: Optional[int] = Field(
        default=30,
        description="回測天數，預設 30 天",
        ge=1,
        le=365,
    )


# ---------------------------------------------------------------------------
# OpenAI function-calling schema (mirrors the Pydantic models above)
# ---------------------------------------------------------------------------

_FUNCTION_SCHEMA: Dict[str, Any] = {
    "name": "parse_stock_query",
    "description": (
        "解析自然語言股票查詢，萃取出意圖類型、技術指標條件、市場範圍和時間範圍。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": [
                    "indicator_screener",
                    "pattern_query",
                    "fundamental_query",
                ],
                "description": (
                    "查詢類型: "
                    "indicator_screener（技術面選股）, "
                    "pattern_query（K線型態查詢）, "
                    "fundamental_query（基本面查詢）"
                ),
            },
            "indicators": {
                "type": "array",
                "description": "技術指標條件列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "技術指標名稱，如 MACD, RSI, MA, KD, BOLL, "
                                "ATR, VWAP, OBV, WR, CCI 等"
                            ),
                        },
                        "comparison": {
                            "type": "string",
                            "enum": [
                                "above",
                                "below",
                                "crossover",
                                "crossunder",
                            ],
                            "description": (
                                "比較方向: above（大於）, below（小於）, "
                                "crossover（黃金交叉/向上穿越）, "
                                "crossunder（死亡交叉/向下穿越）"
                            ),
                        },
                        "value": {
                            "type": "number",
                            "description": (
                                "閾值數值（可選）。例如 RSI < 30 時 value=30。"
                                "交叉型條件通常不需要此欄位。"
                            ),
                        },
                        "params": {
                            "type": "object",
                            "description": (
                                "指標參數。例如 MACD 的 fast/slow/signal，"
                                "MA 的 period，RSI 的 period 等。"
                                "若使用者未指定則留空物件。"
                            ),
                            "additionalProperties": True,
                        },
                    },
                    "required": ["name", "comparison"],
                },
            },
            "stock_scope": {
                "type": "string",
                "description": (
                    "股票市場範圍，如 A股、美股、港股、加密貨幣。"
                    "預設 A股。"
                ),
                "default": "A股",
            },
            "time_range": {
                "type": "integer",
                "description": (
                    "回測/觀察天數。預設 30。例如「近 30 天」= 30，"
                    "「近一週」= 7，「近一個月」= 30。"
                ),
                "minimum": 1,
                "maximum": 365,
                "default": 30,
            },
        },
        "required": ["intent_type", "indicators", "stock_scope"],
    },
}

# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一個專業的金融技術分析助手，專門解析中文或英文的自然語言股票篩選查詢。
你的任務是將使用者的自由文字轉換為結構化的意圖 (QueryIntent)。

## 解析規則

### 1. 意圖判斷 (intent_type)
- **indicator_screener**: 使用者想根據技術指標條件篩選股票
  （例：「找出 MACD 黃金交叉的股票」「RSI < 30 的 A 股」）
- **pattern_query**: 使用者在詢問 K 線型態、圖表型態
  （例：「出現頭肩底的股票」「連續三天紅K」）
- **fundamental_query**: 使用者在詢問基本面資訊
  （例：「營收成長超過 20% 的公司」「PE ratio < 15」）

### 2. 技術指標 (indicators)
對於每個指標條件：
- **name**: 指標名稱（大寫標準縮寫），支援但不限於：
  MACD, RSI, MA, EMA, SMA, KD, KDJ, BOLL, ATR, VWAP, OBV, WR, CCI, DMI, ADX, TRIX
- **comparison**:
  - `above`: 指標值大於某數值（「大於」「超過」「高於」）
  - `below`: 指標值小於某數值（「小於」「低於」）
  - `crossover`: 短期線上穿長期線 / 黃金交叉（「黃金交叉」「上穿」「突破」）
  - `crossunder`: 短期線下穿長期線 / 死亡交叉（「死亡交叉」「下穿」）
- **value**: 閾值數值（若適用）
- **params**: 指標參數。若使用者有指定參數則填入，否則留空物件 {}
  - MACD 預設: fast=12, slow=26, signal=9
  - RSI 預設: period=14
  - MA/EMA: period=20 / period=60 等
  - KD: k_period=9, d_period=3

### 3. 市場範圍 (stock_scope)
- 從文字中辨認市場：A股、美股、港股、加密貨幣 等
- 若無明確指定，預設為 "A股"

### 4. 時間範圍 (time_range)
- 「近 N 天」/ 「過去 N 天」 → N
- 「近一週」/ 「最近一週」 → 7
- 「近一個月」/ 「近一個月」 → 30
- 「近三個月」 → 90
- 若無明確指定，預設為 30

## 輸出要求
- 只輸出 function call，不要自行回答問題
- 所有欄位都要填寫（value 和 params 若無則填 null / {{}})
- 若有多個指標條件，全部列在 indicators 陣列中
- comparison 必須為上述四個值之一
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_extraction_prompt() -> str:
    """Return the system prompt used for LLM-based query extraction.

    Useful for debugging, logging, or displaying the prompt in a UI.

    Returns:
        The full system prompt string.
    """
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def parse_query(user_input: str, *, model: str = "gpt-4o-mini") -> QueryIntent:
    """Parse a natural-language stock query into a structured ``QueryIntent``.

    Args:
        user_input: Free-form text from the user, e.g.
            ``"找出 MACD 黃金交叉且 RSI < 30 的 A 股，近 30 天"``.
        model: OpenAI model to use (default ``"gpt-4o-mini"``).

    Returns:
        A validated :class:`QueryIntent` instance.

    Raises:
        ValueError: If the LLM response cannot be parsed or validated.
        RuntimeError: If the OpenAI API call fails.
    """
    if not user_input or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    client = OpenAI()  # 透過環境變數 OPENAI_API_KEY 讀取金鑰

    # 呼叫 OpenAI function calling
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            tools=[{"type": "function", "function": _FUNCTION_SCHEMA}],
            tool_choice={"type": "function", "function": {"name": "parse_stock_query"}},
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

    logger.debug("LLM 原始回傳參數: %s", json.dumps(raw_args, ensure_ascii=False))

    # 使用 Pydantic 驗證並建構 QueryIntent
    try:
        intent = QueryIntent.model_validate(raw_args)
    except Exception as exc:
        raise ValueError(
            f"LLM output failed Pydantic validation: {exc}"
        ) from exc

    return intent


# ---------------------------------------------------------------------------
# Convenience: module-level demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 快速測試（需設定 OPENAI_API_KEY 環境變數）
    sample_queries = [
        "找出 MACD 黃金交叉且 RSI < 30 的 A 股，近 30 天",
        "美股有哪些股票 KD 值出現死亡交叉？",
        "近一週 A 股布林通道突破上軌的股票",
        "加密貨幣中 EMA 20 上穿 EMA 60 的標的",
    ]
    for q in sample_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        try:
            result = parse_query(q)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
