"""
Code Generator
==============

LLM-based strategy code generator that takes a structured ``StrategyIntent``
and produces executable Python strategy code.

Pipeline:
1. Build a detailed prompt from the ``StrategyIntent``
2. Call the LLM (OpenAI-compatible API) to generate a strategy class
3. Validate the generated code with ``CodeValidator`` (syntax, safety, interface)
4. If validation fails, retry with error feedback (up to 3 attempts)
5. Return the validated code string

The generated code always follows a standard interface::

    import pandas as pd

    class MyStrategy(StrategyBase):
        def __init__(self) -> None:
            self.params = {"period": 14, "threshold": 70}

        def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
            df = data.copy()
            df["signal"] = 0
            # ... strategy logic ...
            return df

Typical usage::

    from nlu.intent_classifier import StrategyIntent, StrategyType

    intent = StrategyIntent(
        strategy_type=StrategyType.MEAN_REVERSION,
        confidence=0.9,
        sub_type="rsi_extreme",
    )
    code = generate_strategy_code(intent)
    exec(code)  # or use CodeValidator + sandbox
"""

from __future__ import annotations

import ast
import logging
import re
from typing import List, Optional, Tuple

from openai import OpenAI

from nlu.intent_classifier import StrategyIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StrategyBase – base class that generated strategies must inherit
# ---------------------------------------------------------------------------


class StrategyBase:
    """Base class for LLM-generated trading strategies.

    Generated strategies should inherit from this class and implement
    the ``generate_signals`` method.

    Attributes:
        params: Dictionary of strategy parameters (indicator periods,
            thresholds, stop-loss percentages, etc.).
    """

    params: dict = {}

    def generate_signals(self, data: "pd.DataFrame") -> "pd.DataFrame":
        """Generate trading signals from OHLCV data.

        Args:
            data: Input DataFrame with columns such as ``open``, ``high``,
                ``low``, ``close``, ``volume``.

        Returns:
            A copy of *data* with an added ``signal`` column where
            ``1`` = buy, ``-1`` = sell, ``0`` = hold.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_signals()"
        )


# ---------------------------------------------------------------------------
# CodeValidator
# ---------------------------------------------------------------------------

# Modules that are explicitly forbidden in generated code.
_DANGEROUS_MODULES: frozenset[str] = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "ctypes",
    "importlib",
    "pickle",
    "shelve",
    "marshal",
    "sqlite3",
    "pathlib",
    "io",
    "tempfile",
    "glob",
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "signal",
    "mmap",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
})

# Function names / attribute accesses that are forbidden.
_DANGEROUS_FUNCTIONS: frozenset[str] = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "exit",
    "quit",
    "input",
})

# Allowed modules (besides builtins).
_ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas",
    "pd",
    "numpy",
    "np",
    "math",
    "statistics",
    "decimal",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "copy",
    "enum",
    "typing",
    "re",
    "json",
    "csv",
    "bisect",
    "heapq",
    "array",
    "abc",
    "dataclasses",
    "warnings",
    "operator",
    "random",
    "string",
    "textwrap",
    "time",
    "calendar",
})


class CodeValidator:
    """Validates LLM-generated Python strategy code for syntax, safety,
    and interface compliance.

    Example::

        validator = CodeValidator()
        ok, errors = validator.validate_all(code)
        if not ok:
            print("Validation failed:", errors)
    """

    # -- Syntax validation ---------------------------------------------------

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, str]:
        """Check whether *code* is syntactically valid Python.

        Uses ``ast.parse`` to perform the check.

        Args:
            code: Python source code string.

        Returns:
            A ``(is_valid, message)`` tuple.  ``message`` is empty on
            success or contains the ``SyntaxError`` details on failure.
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    # -- Safety validation ---------------------------------------------------

    @staticmethod
    def validate_safety(code: str) -> Tuple[bool, str]:
        """Check that *code* does not contain dangerous operations.

        Forbidden:
        * Imports of ``os``, ``sys``, ``subprocess``, ``shutil``, ``socket``,
          ``http``, ``urllib``, ``requests``, and other system-level modules.
        * Calls to ``exec``, ``eval``, ``__import__``, ``open``, ``globals``,
          ``locals``, ``setattr``, ``getattr``, ``compile``, etc.
        * File I/O or network operations.

        Only ``pandas``, ``numpy``, ``math``, and Python builtins are
        allowed.

        Args:
            code: Python source code string.

        Returns:
            ``(is_valid, message)`` where *message* describes the first
            violation found, or is empty if the code is safe.
        """
        errors: List[str] = []

        # --- Check import statements via AST --------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Not our job here; validate_syntax handles this.
            return True, ""

        for node in ast.walk(tree):
            # import X / import X as Y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level in _DANGEROUS_MODULES:
                        errors.append(
                            f"Forbidden import: '{alias.name}' "
                            f"(line {node.lineno})"
                        )
                    elif top_level not in _ALLOWED_MODULES:
                        errors.append(
                            f"Disallowed import: '{alias.name}' "
                            f"(line {node.lineno}) – only pandas/numpy/math "
                            f"and standard library data modules are permitted"
                        )

            # from X import Y
            if isinstance(node, ast.ImportFrom) and node.module:
                top_level = node.module.split(".")[0]
                if top_level in _DANGEROUS_MODULES:
                    errors.append(
                        f"Forbidden import from '{node.module}' "
                        f"(line {node.lineno})"
                    )
                elif top_level not in _ALLOWED_MODULES:
                    errors.append(
                        f"Disallowed import from '{node.module}' "
                        f"(line {node.lineno}) – only pandas/numpy/math "
                        f"and standard library data modules are permitted"
                    )

        # --- Check dangerous function calls / attribute access via AST ------
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = _get_call_name(node)
                if func_name in _DANGEROUS_FUNCTIONS:
                    errors.append(
                        f"Forbidden function call: '{func_name}()' "
                        f"(line {node.lineno})"
                    )
            if isinstance(node, ast.Attribute):
                chain = _get_attribute_chain(node)
                # Block patterns like os.system, os.path, etc.
                parts = chain.split(".")
                if len(parts) >= 2 and parts[0] in _DANGEROUS_MODULES:
                    errors.append(
                        f"Forbidden attribute access: '{chain}' "
                        f"(line {node.lineno})"
                    )

        # --- Regex fallback: catch patterns that AST might miss --------------
        _BLOCKED_REGEX: List[Tuple[str, str]] = [
            (r"\bopen\s*\(", "File I/O: open()"),
            (r"\.write\s*\(", "File I/O: .write()"),
            (r"\.read\s*\(", "File I/O: .read()"),
            (r"\bexec\s*\(", "Dynamic: exec()"),
            (r"\beval\s*\(", "Dynamic: eval()"),
            (r"\b__import__\s*\(", "Dynamic: __import__()"),
            (r"\binput\s*\(", "Interactive: input()"),
        ]
        for line_num, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, description in _BLOCKED_REGEX:
                if re.search(pattern, line):
                    errors.append(f"{description} (line {line_num})")

        if errors:
            return False, "; ".join(sorted(set(errors)))
        return True, ""

    # -- Interface validation ------------------------------------------------

    @staticmethod
    def validate_interface(code: str) -> Tuple[bool, str]:
        """Check that *code* defines a class with a ``generate_signals``
        method that accepts ``(self, data)``.

        Args:
            code: Python source code string.

        Returns:
            ``(is_valid, message)`` describing the first structural
            issue found, or empty string if the interface is correct.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, "Cannot parse code (syntax error)"

        class_name: Optional[str] = None
        has_generate_signals: bool = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if class_name is None:
                    class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "generate_signals":
                            has_generate_signals = True
                            # Verify (self, data) signature
                            args = item.args
                            if len(args.args) < 2:
                                return (
                                    False,
                                    "generate_signals must accept "
                                    "(self, data) parameters",
                                )

        if class_name is None:
            return False, "No class definition found in generated code"
        if not has_generate_signals:
            return (
                False,
                "No 'generate_signals(self, data)' method found "
                "in strategy class",
            )
        return True, ""

    # -- Combined validation -------------------------------------------------

    def validate_all(self, code: str) -> Tuple[bool, List[str]]:
        """Run all validation checks and collect errors.

        Args:
            code: Python source code string.

        Returns:
            ``(all_passed, errors)`` where *errors* is a list of
            error messages (empty when *all_passed* is ``True``).
        """
        errors: List[str] = []

        ok, msg = self.validate_syntax(code)
        if not ok:
            errors.append(f"[syntax] {msg}")

        ok, msg = self.validate_safety(code)
        if not ok:
            errors.append(f"[safety] {msg}")

        ok, msg = self.validate_interface(code)
        if not ok:
            errors.append(f"[interface] {msg}")

        return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# AST helper utilities
# ---------------------------------------------------------------------------


def _get_call_name(node: ast.Call) -> str:
    """Return the simple name of a function being called."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _get_attribute_chain(node: ast.Attribute) -> str:
    """Build a dotted attribute chain such as ``os.system``."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


# ---------------------------------------------------------------------------
# extract_class_name
# ---------------------------------------------------------------------------


def extract_class_name(code: str) -> str:
    """Extract the first class name from *code* using the AST.

    Args:
        code: Python source code string.

    Returns:
        The name of the first ``class`` definition found.

    Raises:
        ValueError: If no class definition is found.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    raise ValueError("No class definition found in the provided code")


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """\
You are an expert Python quantitative trading strategy code generator.
You output ONLY valid Python code — no explanations, no markdown fences,
no prose.

Rules:
1. Define a single class that inherits from `StrategyBase`.
2. The class MUST implement `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame`.
3. `generate_signals` must return a DataFrame with a 'signal' column:
   1 = buy, -1 = sell, 0 = hold.
4. Store all tuneable parameters in `self.params` (a dict).
5. You may only use `pandas`, `numpy`, and `math`. No other imports.
6. No file I/O, no network calls, no exec/eval/open.
7. Include risk-management rules (e.g. stop-loss, position limits) in the
   signal logic when the intent describes them.
8. Add concise docstrings (English) to the class and the method.
"""

_EXAMPLE_CODE: str = """\
Example strategy (RSI oversold/overbought):

```python
import pandas as pd
import numpy as np

class RSIMeanReversion(StrategyBase):
    \"\"\"Buy when RSI falls below oversold, sell when above overbought.\"\"\"

    def __init__(self) -> None:
        self.params = {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
        }

    def _calc_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate buy/sell/hold signals based on RSI.\"\"\"
        df = data.copy()
        df["rsi"] = self._calc_rsi(df["close"], self.params["rsi_period"])
        df["signal"] = 0
        df.loc[df["rsi"] < self.params["oversold"], "signal"] = 1
        df.loc[df["rsi"] > self.params["overbought"], "signal"] = -1
        return df
```
"""


def _build_prompt(intent: StrategyIntent) -> str:
    """Build the user-facing prompt for the LLM.

    Args:
        intent: The parsed strategy intent.

    Returns:
        A prompt string describing the desired strategy.
    """
    parts = [
        "Generate a complete Python strategy class for the following spec.",
        "",
        f"Strategy type : {intent.strategy_type.value}",
        f"Sub-type      : {intent.sub_type}",
        f"Confidence    : {intent.confidence}",
        "",
        "Requirements:",
        "- The class inherits from StrategyBase.",
        "- Implement generate_signals(self, data: pd.DataFrame) -> pd.DataFrame.",
        "- Return a DataFrame with a 'signal' column (1=buy, -1=sell, 0=hold).",
        "- Store tuneable parameters in self.params (dict).",
        "- Use only pandas, numpy, and math.",
        "- Include relevant risk management (stop-loss, etc.) when applicable.",
        "- Add English docstrings to the class and method.",
        "",
        _EXAMPLE_CODE,
        "",
        "Output the complete Python code now.  Do NOT wrap it in markdown fences.",
    ]
    return "\n".join(parts)


def _extract_code_block(response: str) -> str:
    """Extract Python code from an LLM response.

    Tries to find a `````python ... ````` block first; falls back to
    treating the entire response as code.

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted Python source code.
    """
    # Try markdown code blocks
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(response)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: strip lines before the first import / class
    lines = response.strip().splitlines()
    code_lines: list[str] = []
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

    return response.strip()


# ---------------------------------------------------------------------------
# generate_strategy_code
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3


def generate_strategy_code(
    intent: StrategyIntent,
    model: str = "gpt-4o-mini",
) -> str:
    """Generate executable Python strategy code from a structured intent.

    Uses an LLM to produce a strategy class, then validates it with
    ``CodeValidator``.  If validation fails the function feeds the error
    messages back to the LLM and retries (up to 3 times total).

    Args:
        intent: Parsed strategy intent (from ``classify_intent``).
        model: OpenAI model name (default ``"gpt-4o-mini"``).

    Returns:
        A validated Python source code string defining a strategy class
        that inherits from ``StrategyBase`` and implements
        ``generate_signals``.

    Raises:
        RuntimeError: If the OpenAI API call fails.
        ValueError: If the generated code fails validation after all
            retry attempts.

    Example::

        from nlu.intent_classifier import StrategyIntent, StrategyType

        intent = StrategyIntent(
            strategy_type=StrategyType.MOMENTUM,
            confidence=0.85,
            sub_type="breakout",
        )
        code = generate_strategy_code(intent)
        print(code)
    """
    client = OpenAI()
    validator = CodeValidator()
    user_prompt = _build_prompt(intent)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    code: str = ""
    errors: List[str] = []

    for attempt in range(1, _MAX_RETRIES + 1):
        logger.info("Code generation attempt %d/%d", attempt, _MAX_RETRIES)

        # If we have errors from a previous attempt, append feedback.
        if errors:
            feedback = (
                "The code you previously generated has the following "
                "errors.  Please fix ALL of them and return the corrected "
                "complete code:\n\n"
                + "\n".join(f"- {e}" for e in errors)
            )
            messages.append({"role": "user", "content": feedback})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        raw = response.choices[0].message.content or ""
        code = _extract_code_block(raw)

        # Validate
        all_ok, errors = validator.validate_all(code)
        if all_ok:
            logger.info("Code validated successfully on attempt %d", attempt)
            return code

        logger.warning(
            "Validation failed on attempt %d: %s",
            attempt,
            "; ".join(errors),
        )

    # All retries exhausted
    raise ValueError(
        f"Generated code failed validation after {_MAX_RETRIES} attempts: "
        + "; ".join(errors)
    )
