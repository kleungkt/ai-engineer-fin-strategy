"""
Code validator module for AI Strategy Generator.

Validates Python code and Backtrader strategy code using AST analysis.
"""

import ast
import re
from typing import Optional


def validate_python_code(code: str) -> tuple[bool, str]:
    """
    Validate that the given code is syntactically correct Python.

    Args:
        code: Python source code string

    Returns:
        Tuple of (is_valid, message). If invalid, message contains the error.
    """
    if not code or not code.strip():
        return False, "Empty code provided"

    try:
        tree = ast.parse(code)
        return True, "Valid Python code"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}"


def validate_backtrader_strategy(code: str) -> tuple[bool, str]:
    """
    Validate that the code contains a valid Backtrader strategy class.

    Checks:
    1. Valid Python syntax
    2. Contains a class that inherits from bt.Strategy (or backtrader.Strategy)
    3. Has a __init__ method
    4. Has a next method

    Args:
        code: Python source code string expected to define a Backtrader strategy

    Returns:
        Tuple of (is_valid, message)
    """
    # First check basic Python syntax
    is_valid, msg = validate_python_code(code)
    if not is_valid:
        return False, f"Invalid Python: {msg}"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Find classes
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if not classes:
        return False, "No class definition found in the code"

    # Check for strategy-like class
    strategy_found = False
    for cls in classes:
        # Check base classes for bt.Strategy / backtrader.Strategy / object fallback
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Attribute):
                base_names.append(base.attr)
            elif isinstance(base, ast.Name):
                base_names.append(base.id)

        # Accept if base is Strategy-like or code imports backtrader
        is_strategy_class = any(
            name in ("Strategy", "btStrategy") for name in base_names
        )

        # Also check if the code imports backtrader and uses indicators
        has_backtrader_import = "backtrader" in code or "import bt" in code
        has_indicators = "self.data" in code or "self.datas" in code or "self.sma" in code or "self.rsi" in code

        if is_strategy_class or (has_backtrader_import and has_indicators):
            # Check for required methods
            method_names = {
                node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)
            }

            if "__init__" not in method_names:
                return False, f"Class '{cls.name}' is missing __init__ method"
            if "next" not in method_names:
                return False, f"Class '{cls.name}' is missing next method"

            strategy_found = True
            break

    if not strategy_found:
        return False, "No valid Backtrader strategy class found (must inherit from bt.Strategy with __init__ and next methods)"

    return True, "Valid Backtrader strategy"


def extract_class_name(code: str) -> Optional[str]:
    """
    Extract the name of the first class defined in the code.

    Args:
        code: Python source code string

    Returns:
        Class name as string, or None if no class found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return node.name
    except SyntaxError:
        pass
    return None
