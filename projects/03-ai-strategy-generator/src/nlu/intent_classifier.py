"""
LLM-based intent classifier for trading strategy descriptions.

Classifies natural language trading strategy descriptions into structured
``StrategyIntent`` objects using OpenAI function calling. Supports both
Chinese and English inputs.

Typical usage::

    intent = classify_intent("RSI 低於 30 時買入，高於 70 時賣出")
    print(intent.strategy_type)  # "mean_reversion"
    print(intent.sub_type)       # "rsi_extreme"
    print(intent.confidence)     # 0.85
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StrategyType(str, Enum):
    """Supported trading strategy types."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    ML_BASED = "ml_based"


class StrategyIntent(BaseModel):
    """Parsed intent from a natural-language trading strategy description.

    Attributes:
        strategy_type: High-level strategy category.
        confidence: Classification confidence score between 0 and 1.
        sub_type: Specific sub-type of the strategy, e.g. ``'breakout'``,
            ``'crossover'``, ``'bollinger_rsi'``, ``'rsi_extreme'``.
    """

    strategy_type: StrategyType = Field(
        ...,
        description=(
            "策略類型: momentum（動量）, mean_reversion（均值回歸）, "
            "trend_following（趨勢跟蹤）, arbitrage（套利）, ml_based（機器學習）"
        ),
    )
    confidence: float = Field(
        ...,
        description="分類置信度，0.0 到 1.0 之間",
        ge=0.0,
        le=1.0,
    )
    sub_type: str = Field(
        ...,
        description=(
            "策略子類型，例如: breakout（突破）, crossover（交叉）, "
            "rsi_extreme（RSI極值）, bollinger_rsi（布林+RSI組合）, "
            "macd_divergence（MACD背離）, trend_channel（趨勢通道）"
        ),
    )

    @field_validator("sub_type")
    @classmethod
    def validate_sub_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("sub_type must be a non-empty string")
        return v.strip().lower()


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

_FUNCTION_SCHEMA: Dict[str, Any] = {
    "name": "classify_strategy_intent",
    "description": (
        "將自然語言交易策略描述分類為結構化的策略意圖，"
        "包括策略類型、置信度和子類型。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "strategy_type": {
                "type": "string",
                "enum": [
                    "momentum",
                    "mean_reversion",
                    "trend_following",
                    "arbitrage",
                    "ml_based",
                ],
                "description": (
                    "策略類型: "
                    "momentum（動量策略 - 追強殺弱）, "
                    "mean_reversion（均值回歸 - 高拋低吸）, "
                    "trend_following（趨勢跟蹤 - 順勢而為）, "
                    "arbitrage（套利 - 價差交易）, "
                    "ml_based（機器學習 - AI模型預測）"
                ),
            },
            "confidence": {
                "type": "number",
                "description": "分類置信度，0.0 到 1.0",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "sub_type": {
                "type": "string",
                "description": (
                    "策略子類型，具體描述策略方法，例如: "
                    "breakout（突破）, crossover（均線交叉）, "
                    "rsi_extreme（RSI極值反轉）, bollinger_rsi（布林+RSI組合）, "
                    "macd_divergence（MACD背離）, trend_channel（趨勢通道）, "
                    "pairs_trading（配對交易）, statistical_arb（統計套利）, "
                    "reinforcement_learning（強化學習）"
                ),
            },
        },
        "required": ["strategy_type", "confidence", "sub_type"],
    },
}

# ---------------------------------------------------------------------------
# System prompt with few-shot examples
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一個專業的量化交易策略分析師，專門將自然語言策略描述分類為結構化意圖。
你支援中文和英文輸入。

## 策略類型定義

### 1. momentum（動量策略）
追蹤價格動量，追強殺弱，順應短期趨勢方向操作。
典型子類型: breakout（突破）, relative_strength（相對強弱）, volume_price（量價配合）

### 2. mean_reversion（均值回歸）
相信價格會回歸均值，在超買/超賣時反向操作。
典型子類型: rsi_extreme（RSI極值反轉）, bollinger_rsi（布林+RSI組合）, z_score（標準分回歸）

### 3. trend_following（趨勢跟蹤）
順應中長期趨勢方向，在趨勢確認後進場。
典型子類型: crossover（均線交叉）, macd_divergence（MACD背離）, trend_channel（趨勢通道）, moving_average_system（均線系統）

### 4. arbitrage（套利）
利用市場價格差異進行低風險套利交易。
典型子類型: pairs_trading（配對交易）, statistical_arb（統計套利）, index_arb（指數套利）

### 5. ml_based（機器學習）
使用機器學習或深度學習模型進行預測和交易。
典型子類型: reinforcement_learning（強化學習）, feature_engineering（特徵工程）, sentiment_analysis（情緒分析）

## 分類規則

1. 仔細分析策略描述中的關鍵詞和邏輯
2. 如果描述涉及多個策略類型，選擇最主要的一個
3. 根據描述的明確程度給出合理的置信度
4. sub_type 要盡量具體地描述策略方法

## Few-shot 示例

### 示例 1: 動量策略
輸入: "當股票突破20日新高且成交量放大時買入，持有到跌破10日均線"
輸出: strategy_type=momentum, sub_type=breakout, confidence=0.9

### 示例 2: 均值回歸
輸入: "RSI低於30時買入，RSI高於70時賣出"
輸出: strategy_type=mean_reversion, sub_type=rsi_extreme, confidence=0.95

### 示例 3: 均值回歸（複合）
輸入: "布林通道下軌且RSI超賣時分批建倉，觸及上軌時減倉"
輸出: strategy_type=mean_reversion, sub_type=bollinger_rsi, confidence=0.9

### 示例 4: 趨勢跟蹤
輸入: "MACD黃金交叉買入，死亡交叉賣出"
輸出: strategy_type=trend_following, sub_type=crossover, confidence=0.85

### 示例 5: 趨勢跟蹤
輸入: "Use a dual moving average crossover system: buy when 20-day EMA crosses above 50-day EMA, sell when it crosses below"
輸出: strategy_type=trend_following, sub_type=crossover, confidence=0.95

### 示例 6: 套利
輸入: "找出相關性高的股票配對，價差偏離均值2倍標準差時做空價差"
輸出: strategy_type=arbitrage, sub_type=pairs_trading, confidence=0.85

### 示例 7: 機器學習
輸入: "用LSTM模型預測股價走勢，結合情緒分析信號進行交易"
輸出: strategy_type=ml_based, sub_type=reinforcement_learning, confidence=0.8

### 示例 8: 動量策略（中文）
輸入: "連續三天漲停的股票，第四天開盤買入"
輸出: strategy_type=momentum, sub_type=breakout, confidence=0.85

## 輸出要求
- 只輸出 function call，不要自行回答問題
- confidence 應反映描述的明確程度：非常明確 > 0.9，明確 0.7-0.9，模糊 0.5-0.7
- sub_type 使用英文小寫底線格式
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_classification_prompt() -> str:
    """Return the system prompt used for intent classification.

    Useful for debugging, logging, or displaying the prompt in a UI.

    Returns:
        The full system prompt string.
    """
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------


def classify_intent(
    text: str,
    *,
    model: str = "gpt-4o-mini",
) -> StrategyIntent:
    """Classify a natural-language trading strategy description.

    Args:
        text: Free-form text describing a trading strategy, e.g.
            ``"RSI 低於 30 時買入，高於 70 時賣出"``.
        model: OpenAI model to use (default ``"gpt-4o-mini"``).

    Returns:
        A validated :class:`StrategyIntent` instance.

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
                "function": {"name": "classify_strategy_intent"},
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
        "LLM 分類結果: %s", json.dumps(raw_args, ensure_ascii=False)
    )

    # 使用 Pydantic 驗證並建構 StrategyIntent
    try:
        intent = StrategyIntent.model_validate(raw_args)
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
    sample_descriptions = [
        "RSI 低於 30 時買入，高於 70 時賣出",
        "當股票突破20日新高且成交量放大時買入",
        "MACD黃金交叉買入，死亡交叉賣出，設置2%止損",
        "布林通道下軌且RSI超賣時分批建倉",
        "Use LSTM to predict stock price movement",
        "找出相關性高的股票配對，價差偏離時做空價差",
    ]
    for desc in sample_descriptions:
        print(f"\n{'='*60}")
        print(f"Description: {desc}")
        try:
            result = classify_intent(desc)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
