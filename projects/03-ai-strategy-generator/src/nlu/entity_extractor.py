"""
LLM-based entity extractor for trading strategy descriptions.

Extracts structured trading entities from natural language descriptions using
OpenAI function calling. Supports indicators, thresholds, timeframes, assets,
actions, and risk management parameters in both Chinese and English.

Typical usage::

    result = extract_entities("當RSI低於30且成交量放大時買入台積電")
    for entity in result.entities:
        print(entity.entity_type, entity.value, entity.numeric_value)
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


class EntityType(str, Enum):
    """Supported trading entity types."""

    INDICATOR = "indicator"
    THRESHOLD = "threshold"
    TIMEFRAME = "timeframe"
    ASSET = "asset"
    ACTION = "action"
    RISK_MANAGEMENT = "risk_management"


class TradingEntity(BaseModel):
    """A single trading entity extracted from natural language.

    Attributes:
        entity_type: Category of the entity.
        value: String representation of the entity value.
        numeric_value: Numeric value if applicable, otherwise ``None``.
        params: Additional parameters associated with the entity,
            e.g. indicator periods, asset details.
    """

    entity_type: EntityType = Field(
        ...,
        description=(
            "實體類型: indicator（技術指標）, threshold（閾值）, "
            "timeframe（時間框架）, asset（資產）, action（操作）, "
            "risk_management（風險管理）"
        ),
    )
    value: str = Field(
        ...,
        description="實體值的字串表示，例如 'RSI', '30', 'daily', 'buy'",
    )
    numeric_value: Optional[float] = Field(
        default=None,
        description="數值型實體的數值，非數值型實體為 null",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "額外參數。例如指標的週期參數 {period: 14}，"
            "資產的市場資訊 {market: 'TW'} 等"
        ),
    )

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("value must be a non-empty string")
        return v.strip()


class ExtractionResult(BaseModel):
    """Result of entity extraction from a trading strategy description.

    Attributes:
        entities: List of extracted trading entities.
        raw_text: The original input text.
    """

    entities: List[TradingEntity] = Field(
        default_factory=list,
        description="從文本中提取的交易實體列表",
    )
    raw_text: str = Field(
        ...,
        description="原始輸入文本",
    )


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

_FUNCTION_SCHEMA: Dict[str, Any] = {
    "name": "extract_trading_entities",
    "description": (
        "從自然語言交易策略描述中提取所有交易相關實體，"
        "包括技術指標、閾值、時間框架、資產、操作和風險管理參數。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "提取到的交易實體列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "indicator",
                                "threshold",
                                "timeframe",
                                "asset",
                                "action",
                                "risk_management",
                            ],
                            "description": (
                                "實體類型: "
                                "indicator（技術指標如MA, RSI, MACD等）, "
                                "threshold（數值閾值如30, 70, 0.02等）, "
                                "timeframe（時間框架如daily, 5min等）, "
                                "asset（資產如股票代碼, 指數, 加密貨幣等）, "
                                "action（操作如buy, sell, hold等）, "
                                "risk_management（風險管理如stop_loss, take_profit等）"
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "實體值的字串表示。"
                                "例如指標名 'RSI'、閾值 '30'、操作 'buy'"
                            ),
                        },
                        "numeric_value": {
                            "type": "number",
                            "description": (
                                "數值型實體的數值。"
                                "例如閾值30 → 30.0，止損2% → 0.02。"
                                "非數值型實體填 null。"
                            ),
                        },
                        "params": {
                            "type": "object",
                            "description": (
                                "額外參數。"
                                "指標: {period: 14}, {fast: 12, slow: 26, signal: 9}"
                                "資產: {market: 'TW', name: '台積電'}"
                                "風險管理: {value: 0.02, type: 'percent'}"
                            ),
                            "additionalProperties": True,
                        },
                    },
                    "required": ["entity_type", "value"],
                },
            },
        },
        "required": ["entities"],
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一個專業的量化交易實體提取器，專門從自然語言策略描述中提取交易相關實體。
你支援中文和英文輸入。

## 實體類型說明

### 1. indicator（技術指標）
常見指標（提取標準英文縮寫）:
- 移動平均線: MA, EMA, SMA, WMA（值: 指標名稱，params: {period: N}）
- 相對強弱指標: RSI（params: {period: 14}）
- 隨機指標: KD, KDJ（params: {k_period: 9, d_period: 3}）
- MACD（params: {fast: 12, slow: 26, signal: 9}）
- 布林通道: BOLL（params: {period: 20, std_dev: 2}）
- 平均真實範圍: ATR（params: {period: 14}）
- 成交量: VOLUME, OBV
- 威廉指標: WR（params: {period: 14}）
- 順勢指標: CCI（params: {period: 20}）
- 趨向指標: DMI, ADX
- 其他: TRIX, VWAP, SAR

中文別名映射:
- 均線/移動平均線 → MA
- 快線/慢線 → 根據上下文可能是 MACD 的 fast/slow
- 布林通道/布林帶 → BOLL
- 成交量/量 → VOLUME
- 隨機指標/KD值 → KD

### 2. threshold（閾值）
- 數值型條件: "RSI低於30" → entity_type=threshold, value="30", numeric_value=30.0
- 百分比型: "2%止損" → entity_type=threshold, value="2%", numeric_value=0.02
- 注意區分指標參數和交易閾值

### 3. timeframe（時間框架）
- 日線/daily, 週線/weekly, 月線/monthly
- 分鐘線: 1min, 5min, 15min, 30min, 60min
- 近N天/過去N天 → timeframe, params: {days: N}
- "20日均線" 中的 20 是 MA 的 period 參數，不是 timeframe

### 4. asset（資產）
- 股票代碼: 2330, AAPL, TSLA
- 指數: 加權指數, S&P 500, NASDAQ
- 加密貨幣: BTC, ETH
- 板塊/類股: 半導體, 科技股
- 中文股票名稱映射: 台積電 → 2330, 鴻海 → 2317

### 5. action（操作）
- 買入/buy, 賣出/sell, 持有/hold
- 加倉/add_position, 減倉/reduce_position
- 分批建倉 → action: "add_position", params: {method: "batch"}
- 做多/long, 做空/short

### 6. risk_management（風險管理）
- 止損/stop_loss（params: {value: N, type: "percent"|"points"}）
- 止盈/take_profit
- 移動止損/trailing_stop
- 風險比例/risk_ratio

## 提取規則

1. 盡可能多地提取實體，不要遺漏
2. 每個實體都要填寫 entity_type 和 value
3. 如果有明確的數值，填寫 numeric_value
4. 相關參數放在 params 中
5. 注意區分指標參數（如 MA 的 period）和交易閾值（如 RSI 的 30）

## 輸出要求
- 只輸出 function call，不要自行回答問題
- 所有實體都要在 entities 陣列中
- 不要重複提取相同的實體
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_extraction_prompt() -> str:
    """Return the system prompt used for entity extraction.

    Useful for debugging, logging, or displaying the prompt in a UI.

    Returns:
        The full system prompt string.
    """
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def extract_entities(
    text: str,
    *,
    model: str = "gpt-4o-mini",
) -> ExtractionResult:
    """Extract trading entities from a natural-language strategy description.

    Args:
        text: Free-form text describing a trading strategy, e.g.
            ``"當RSI低於30且成交量放大時買入台積電"``.
        model: OpenAI model to use (default ``"gpt-4o-mini"``).

    Returns:
        A validated :class:`ExtractionResult` instance containing all
        extracted entities.

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
                "function": {"name": "extract_trading_entities"},
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
        "LLM 提取結果: %s", json.dumps(raw_args, ensure_ascii=False)
    )

    # 注入原始文本
    raw_args["raw_text"] = text

    # 使用 Pydantic 驗證並建構 ExtractionResult
    try:
        result = ExtractionResult.model_validate(raw_args)
    except Exception as exc:
        raise ValueError(
            f"LLM output failed Pydantic validation: {exc}"
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Convenience: module-level demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 快速測試（需設定 OPENAI_API_KEY 環境變數）
    sample_descriptions = [
        "當RSI低於30且成交量放大時買入台積電",
        "布林通道下軌且RSI超賣時分批建倉",
        "跌破20日均線賣出，設置2%止損",
        "MACD黃金交叉買入，死亡交叉賣出",
        "Buy AAPL when EMA20 crosses above EMA50 with 3% stop loss",
    ]
    for desc in sample_descriptions:
        print(f"\n{'='*60}")
        print(f"Description: {desc}")
        try:
            result = extract_entities(desc)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
