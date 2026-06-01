"""
Code Generator
==============

LLM-based strategy code generator that takes parsed NLU output and generates
complete, runnable Python strategy code.

Pipeline:
1. Build a detailed prompt from intent, entities, and logic
2. Use few-shot examples of complete strategy classes
3. Call LLM to generate strategy code
4. Validate with AST validator
5. Check safety with safety checker
6. Return structured GeneratedStrategy result
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

from nlu.entity_extractor import ExtractionResult
from nlu.intent_classifier import StrategyIntent
from nlu.logic_parser import Condition, TradingLogic

from .ast_validator import ValidationResult, validate_code
from .safety_checker import SafetyReport, check_safety

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GeneratedStrategy(BaseModel):
    """A complete strategy generated from NLU output.

    Attributes:
        name: Strategy name (e.g. "Momentum Breakout v1").
        description: Human-readable strategy description.
        code: Complete, runnable Python source code.
        params: Extracted parameters (indicator periods, thresholds, etc.).
        strategy_type: Strategy category from intent classification.
        risk_rules: List of risk management rule descriptions.
    """

    name: str = Field(..., description="策略名稱")
    description: str = Field(..., description="策略描述")
    code: str = Field(..., description="完整的 Python 策略代碼")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="策略參數字典"
    )
    strategy_type: str = Field(..., description="策略類型")
    risk_rules: List[str] = Field(
        default_factory=list, description="風險管理規則列表"
    )


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: str = """\
## 完整策略代碼範例

### 範例 1: RSI 極值策略

需求: RSI低於30時買入，高於70時賣出

```python
import pandas as pd
import numpy as np


class RSIExtreme:
    \"\"\"RSI 极端值策略

    当RSI低于超卖阈值时买入，高于超买阈值时卖出。
    \"\"\"

    def __init__(self):
        self.params = {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
        }

    def _calc_rsi(self, series: pd.Series, period: int) -> pd.Series:
        \"\"\"计算RSI指标\"\"\"
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"生成交易信号\"\"\"
        df = data.copy()
        df["rsi"] = self._calc_rsi(df["close"], self.params["rsi_period"])
        df["signal"] = 0
        df.loc[df["rsi"] < self.params["oversold"], "signal"] = 1
        df.loc[df["rsi"] > self.params["overbought"], "signal"] = -1
        return df
```

### 範例 2: 動量突破策略

需求: 收盤價突破20日高點時買入，跌破10日低點時賣出，2%止損

```python
import pandas as pd
import numpy as np


class MomentumBreakout:
    \"\"\"动量突破策略

    当价格突破N日最高价时买入，跌破M日最低价时卖出。
    使用ATR进行动态止损。
    \"\"\"

    def __init__(self):
        self.params = {
            "entry_period": 20,
            "exit_period": 10,
            "stop_loss_pct": 0.02,
            "atr_period": 14,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"生成交易信号\"\"\"
        df = data.copy()
        high_n = df["high"].rolling(window=self.params["entry_period"]).max()
        low_n = df["low"].rolling(window=self.params["exit_period"]).min()

        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": abs(df["high"] - df["close"].shift(1)),
            "lc": abs(df["low"] - df["close"].shift(1)),
        }).max(axis=1)
        atr = tr.rolling(window=self.params["atr_period"]).mean()

        df["signal"] = 0
        df.loc[df["close"] > high_n.shift(1), "signal"] = 1
        df.loc[df["close"] < low_n.shift(1), "signal"] = -1
        df["atr"] = atr
        df["stop_loss_distance"] = atr * 2.0
        return df
```
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_conditions(conditions: List[Condition], label: str) -> str:
    """Format a list of conditions into a readable prompt section."""
    if not conditions:
        return f"  {label}: (none)\n"

    lines = [f"  {label}:\n"]
    for i, cond in enumerate(conditions, 1):
        conn = f" {cond.conjunction.value}" if cond.conjunction else ""
        lines.append(
            f"    {i}. {cond.left_operand} {cond.operator.value} "
            f"{cond.right_operand}{conn}\n"
        )
    return "".join(lines)


def _build_generation_prompt(
    intent: StrategyIntent,
    entities: ExtractionResult,
    logic: TradingLogic,
) -> str:
    """Build a detailed prompt for strategy code generation.

    Combines intent classification, entity extraction, and trading logic
    into a comprehensive prompt with few-shot examples.

    Args:
        intent: Classified strategy intent.
        entities: Extracted trading entities.
        logic: Parsed trading logic.

    Returns:
        Complete prompt string for the LLM.
    """
    # 整理实体信息
    indicators: List[str] = []
    thresholds: Dict[str, float] = {}
    timeframes: List[str] = []
    risk_mgmt: List[str] = []

    for entity in entities.entities:
        if entity.entity_type.value == "indicator":
            period = entity.params.get("period", "")
            ind_str = entity.value
            if period:
                ind_str += f" (period={period})"
            indicators.append(ind_str)
        elif entity.entity_type.value == "threshold":
            thresholds[entity.value] = entity.numeric_value or 0.0
        elif entity.entity_type.value == "timeframe":
            timeframes.append(entity.value)
        elif entity.entity_type.value == "risk_management":
            risk_mgmt.append(entity.value)

    # 整理风险规则
    risk_rules_text: List[str] = []
    for rule in logic.risk_rules:
        risk_rules_text.append(
            f"{rule.left_operand} {rule.operator.value} {rule.right_operand}"
        )

    prompt_parts = [
        "你是一個專業的量化交易策略代碼生成器。",
        "根據以下策略需求，生成一個完整的、可執行的 Python 策略類。",
        "",
        "## 策略需求",
        "",
        f"### 意圖分析",
        f"  - 策略類型: {intent.strategy_type.value}",
        f"  - 子類型: {intent.sub_type}",
        f"  - 置信度: {intent.confidence}",
        "",
        "### 提取的實體",
        f"  - 技術指標: {', '.join(indicators) if indicators else '(none)'}",
        f"  - 閾值: {json.dumps(thresholds, ensure_ascii=False) if thresholds else '(none)'}",
        f"  - 時間框架: {', '.join(timeframes) if timeframes else '(default)'}",
        f"  - 風險管理: {', '.join(risk_mgmt) if risk_mgmt else '(none)'}",
        "",
        "### 交易邏輯",
        _format_conditions(logic.entry_conditions, "進場條件"),
        _format_conditions(logic.exit_conditions, "出場條件"),
        f"  倉位管理: {logic.position_sizing.value}",
        _format_conditions(logic.risk_rules, "風險規則"),
        "",
        _FEW_SHOT_EXAMPLES,
        "",
        "## 代碼生成要求",
        "",
        "1. **完整的 Python 類**: 包含 import、class 定義、__init__、generate_signals",
        "2. **標準接口**: class 必須有 `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame`",
        "3. **信號列**: 返回的 DataFrame 必須包含 'signal' 列 (1=買入, -1=賣出, 0=持有)",
        "4. **參數化**: 所有可調參數放在 self.params 字典中",
        "5. **使用 pandas 和 numpy**: 不要使用其他第三方庫",
        "6. **中文注釋**: 使用中文撰寫關鍵邏輯的注釋",
        "7. **類型提示**: 為方法簽名添加類型提示",
        "8. **docstring**: 為類和主要方法撰寫 docstring",
        "",
        "## 輸出格式",
        "",
        "只輸出 Python 代碼，不要包含任何其他文字說明。",
        "代碼必須可以直接被 `exec()` 執行。",
        "",
        "```python",
        "# 你的策略代碼",
        "```",
    ]

    return "\n".join(prompt_parts)


def _build_refinement_prompt(
    strategy: GeneratedStrategy,
    feedback: str,
) -> str:
    """Build a prompt for strategy refinement.

    Args:
        strategy: The current strategy to refine.
        feedback: User feedback on the strategy.

    Returns:
        Refinement prompt string.
    """
    return f"""\
你是一個專業的量化交易策略代碼生成器。
請根據以下反饋，改進現有的策略代碼。

## 現有策略

名稱: {strategy.name}
描述: {strategy.description}
類型: {strategy.strategy_type}

```python
{strategy.code}
```

## 用戶反饋

{feedback}

## 改進要求

1. 保持標準接口: `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame`
2. 保持 'signal' 列 (1=買入, -1=賣出, 0=持有)
3. 只使用 pandas 和 numpy
4. 根據反饋進行修改
5. 使用中文注釋

## 輸出格式

只輸出改進後的完整 Python 代碼，不要包含其他文字。

```python
# 改進後的策略代碼
```
"""


# ---------------------------------------------------------------------------
# Code extraction from LLM response
# ---------------------------------------------------------------------------


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response.

    Handles various formats:
    - Markdown code blocks (```python ... ```)
    - Raw Python code
    - Mixed response with explanation and code

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted Python code string.
    """
    # 尝试提取 markdown 代码块
    code_block_pattern = re.compile(
        r"```(?:python)?\s*\n(.*?)```", re.DOTALL
    )
    matches = code_block_pattern.findall(response)

    if matches:
        # 取最长的代码块（通常是最完整的策略）
        return max(matches, key=len).strip()

    # 如果没有代码块，尝试直接使用响应
    # 去除可能的非代码行
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            in_code = True
        if stripped.startswith("```"):
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    # 最后的 fallback: 直接返回整个响应
    return response.strip()


def _extract_params_from_entities(entities: ExtractionResult) -> Dict[str, Any]:
    """Extract strategy parameters from entities.

    Args:
        entities: Extracted trading entities.

    Returns:
        Dictionary of parameter name -> value.
    """
    params: Dict[str, Any] = {}

    for entity in entities.entities:
        if entity.entity_type.value == "indicator":
            # 提取指标参数
            for key, val in entity.params.items():
                param_name = f"{entity.value.lower()}_{key}"
                params[param_name] = val
        elif entity.entity_type.value == "threshold":
            if entity.numeric_value is not None:
                params[f"threshold_{entity.value}"] = entity.numeric_value
        elif entity.entity_type.value == "risk_management":
            if entity.numeric_value is not None:
                params[entity.value.lower()] = entity.numeric_value

    return params


def _generate_strategy_name(
    intent: StrategyIntent,
    entities: ExtractionResult,
) -> str:
    """Generate a human-readable strategy name.

    Args:
        intent: Strategy intent classification.
        entities: Extracted trading entities.

    Returns:
        Strategy name string.
    """
    # 策略类型中文映射
    type_names = {
        "momentum": "動量",
        "mean_reversion": "均值回歸",
        "trend_following": "趨勢跟蹤",
        "arbitrage": "套利",
        "ml_based": "機器學習",
    }

    type_cn = type_names.get(intent.strategy_type.value, intent.strategy_type.value)
    sub_type = intent.sub_type.replace("_", " ").title()

    # 提取主要指标
    indicators = [
        e.value for e in entities.entities
        if e.entity_type.value == "indicator"
    ]
    indicator_str = "-".join(indicators[:2]) if indicators else ""

    name_parts = [type_cn, sub_type]
    if indicator_str:
        name_parts.append(indicator_str)
    name_parts.append("v1")

    return " ".join(name_parts)


def _generate_description(
    intent: StrategyIntent,
    entities: ExtractionResult,
    logic: TradingLogic,
) -> str:
    """Generate a human-readable strategy description.

    Args:
        intent: Strategy intent.
        entities: Extracted entities.
        logic: Trading logic.

    Returns:
        Strategy description string.
    """
    parts: List[str] = []

    # 策略类型
    parts.append(f"策略類型: {intent.strategy_type.value} ({intent.sub_type})")

    # 进场条件
    if logic.entry_conditions:
        entry_desc = "進場: " + " 且 ".join(
            f"{c.left_operand} {c.operator.value} {c.right_operand}"
            for c in logic.entry_conditions
        )
        parts.append(entry_desc)

    # 出场条件
    if logic.exit_conditions:
        exit_desc = "出場: " + " 且 ".join(
            f"{c.left_operand} {c.operator.value} {c.right_operand}"
            for c in logic.exit_conditions
        )
        parts.append(exit_desc)

    # 风险规则
    if logic.risk_rules:
        risk_desc = "風控: " + ", ".join(
            f"{r.left_operand} {r.operator.value} {r.right_operand}"
            for r in logic.risk_rules
        )
        parts.append(risk_desc)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def generate_strategy(
    intent: StrategyIntent,
    entities: ExtractionResult,
    logic: TradingLogic,
    model: str = "gpt-4o",
) -> GeneratedStrategy:
    """Generate a complete Python strategy from parsed NLU output.

    Builds a detailed prompt combining intent, entities, and logic,
    then uses the LLM with few-shot examples to generate a complete,
    runnable Python strategy class.

    The generated code follows the standard interface:
    - A class with ``generate_signals(self, data: pd.DataFrame) -> pd.DataFrame``
    - Returns a DataFrame with a 'signal' column (1=buy, -1=sell, 0=hold)
    - Parameters stored in ``self.params`` dict

    Args:
        intent: Classified strategy intent from NLU.
        entities: Extracted trading entities from NLU.
        logic: Parsed trading logic from NLU.
        model: OpenAI model to use (default: "gpt-4o").

    Returns:
        GeneratedStrategy with name, description, code, params, etc.

    Raises:
        ValueError: If generated code fails validation.
        RuntimeError: If the OpenAI API call fails.

    Example:
        >>> intent = StrategyIntent(
        ...     strategy_type="mean_reversion",
        ...     sub_type="rsi_extreme",
        ...     confidence=0.9,
        ... )
        >>> strategy = generate_strategy(intent, entities, logic)
        >>> print(strategy.name)
        >>> print(strategy.code)
    """
    # 构建 prompt
    prompt = _build_generation_prompt(intent, entities, logic)

    # 调用 LLM
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一個專業的 Python 量化交易策略代碼生成器。"
                        "你只輸出 Python 代碼，不包含其他解釋文字。"
                        "生成的代碼必須遵循標準策略接口。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 低温度确保代码质量
            max_tokens=4096,
        )
    except Exception as exc:
        logger.error("OpenAI API 呼叫失敗: %s", exc)
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    raw_response = response.choices[0].message.content or ""
    code = _extract_code(raw_response)

    # AST 验证
    validation: ValidationResult = validate_code(code)
    if not validation.is_valid:
        logger.warning(
            "生成的代碼驗證失敗: %s", "; ".join(validation.errors)
        )
        # 尝试一次修复
        code = _attempt_fix(client, code, validation, model)
        validation = validate_code(code)
        if not validation.is_valid:
            raise ValueError(
                f"Generated code validation failed after fix attempt: "
                f"{'; '.join(validation.errors)}"
            )

    # 安全检查
    safety: SafetyReport = check_safety(code)
    if not safety.is_safe:
        raise ValueError(
            f"Generated code failed safety check: "
            f"{'; '.join(safety.blocked_patterns)}"
        )

    # 构建结果
    name = _generate_strategy_name(intent, entities)
    description = _generate_description(intent, entities, logic)
    params = _extract_params_from_entities(entities)

    # 合并逻辑中的参数
    for cond in logic.entry_conditions + logic.exit_conditions:
        try:
            val = float(cond.right_operand)
            param_key = cond.left_operand.lower().replace("(", "_").replace(")", "")
            params[param_key] = val
        except ValueError:
            pass

    risk_rules = [
        f"{r.left_operand} {r.operator.value} {r.right_operand}"
        for r in logic.risk_rules
    ]

    return GeneratedStrategy(
        name=name,
        description=description,
        code=code,
        params=params,
        strategy_type=intent.strategy_type.value,
        risk_rules=risk_rules,
    )


def _attempt_fix(
    client: OpenAI,
    code: str,
    validation: ValidationResult,
    model: str,
) -> str:
    """Attempt to fix validation errors in generated code.

    Args:
        client: OpenAI client instance.
        code: The code with validation errors.
        validation: Validation result with errors.
        model: Model to use for the fix.

    Returns:
        Fixed code string.
    """
    errors_text = "\n".join(f"- {e}" for e in validation.errors)
    warnings_text = "\n".join(f"- {w}" for w in validation.warnings)

    fix_prompt = f"""\
以下 Python 策略代碼有錯誤，請修復：

```python
{code}
```

錯誤:
{errors_text}

警告:
{warnings_text}

要求:
1. 保持 generate_signals 接口
2. 修復所有錯誤
3. 只輸出修復後的完整代碼
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "修復 Python 策略代碼中的錯誤，只輸出修復後的代碼。",
                },
                {"role": "user", "content": fix_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
        return _extract_code(raw)
    except Exception as exc:
        logger.error("代碼修復失敗: %s", exc)
        return code


def refine_strategy(
    strategy: GeneratedStrategy,
    feedback: str,
    model: str = "gpt-4o",
) -> GeneratedStrategy:
    """Iteratively refine a strategy based on user feedback.

    Takes an existing GeneratedStrategy and user feedback, then uses the LLM
    to produce an improved version.

    Args:
        strategy: The current strategy to refine.
        feedback: User feedback describing desired changes.
        model: OpenAI model to use (default: "gpt-4o").

    Returns:
        A new GeneratedStrategy with refined code.

    Raises:
        ValueError: If refined code fails validation.
        RuntimeError: If the OpenAI API call fails.

    Example:
        >>> strategy = generate_strategy(intent, entities, logic)
        >>> refined = refine_strategy(strategy, "增加2%的止损逻辑")
        >>> print(refined.code)
    """
    prompt = _build_refinement_prompt(strategy, feedback)

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一個專業的量化交易策略代碼生成器。"
                        "根據用戶反饋改進策略代碼，只輸出 Python 代碼。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.error("OpenAI API 呼叫失敗: %s", exc)
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    raw_response = response.choices[0].message.content or ""
    new_code = _extract_code(raw_response)

    # 验证修复后的代码
    validation = validate_code(new_code)
    if not validation.is_valid:
        new_code = _attempt_fix(client, new_code, validation, model)
        validation = validate_code(new_code)
        if not validation.is_valid:
            raise ValueError(
                f"Refined code validation failed: "
                f"{'; '.join(validation.errors)}"
            )

    # 安全检查
    safety = check_safety(new_code)
    if not safety.is_safe:
        raise ValueError(
            f"Refined code failed safety check: "
            f"{'; '.join(safety.blocked_patterns)}"
        )

    # 构建新的策略（保留原有元数据，更新代码和描述）
    return GeneratedStrategy(
        name=f"{strategy.name} (refined)",
        description=f"{strategy.description} [refined: {feedback}]",
        code=new_code,
        params=strategy.params,
        strategy_type=strategy.strategy_type,
        risk_rules=strategy.risk_rules,
    )
