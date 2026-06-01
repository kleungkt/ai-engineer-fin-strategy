"""Tests for the code_validator module."""

import pytest
from code_validator import validate_python_code, validate_backtrader_strategy, extract_class_name


class TestValidatePythonCode:
    """Tests for validate_python_code()."""

    def test_valid_simple_code(self):
        code = "x = 1\nprint(x)"
        is_valid, msg = validate_python_code(code)
        assert is_valid is True
        assert "Valid" in msg

    def test_valid_function_def(self):
        code = "def foo():\n    return 42"
        is_valid, msg = validate_python_code(code)
        assert is_valid is True

    def test_empty_code(self):
        is_valid, msg = validate_python_code("")
        assert is_valid is False
        assert "Empty" in msg

    def test_whitespace_only(self):
        is_valid, msg = validate_python_code("   \n  \n  ")
        assert is_valid is False
        assert "Empty" in msg

    def test_syntax_error(self):
        code = "def foo(:\n    pass"
        is_valid, msg = validate_python_code(code)
        assert is_valid is False
        assert "Syntax error" in msg

    def test_missing_colon(self):
        code = "if True\n    pass"
        is_valid, msg = validate_python_code(code)
        assert is_valid is False


class TestValidateBacktraderStrategy:
    """Tests for validate_backtrader_strategy()."""

    def test_valid_strategy(self):
        code = """
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
"""
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is True
        assert "Valid" in msg

    def test_valid_strategy_with_object_base(self):
        """Code that has backtrader import + indicators but uses bt.Strategy."""
        code = """
import backtrader as bt

class MyStrat(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=20)

    def next(self):
        pass
"""
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is True

    def test_no_class(self):
        code = "import backtrader as bt\nprint('hello')"
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is False
        assert "No class" in msg

    def test_missing_init(self):
        code = """
import backtrader as bt

class MyStrategy(bt.Strategy):
    def next(self):
        pass
"""
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is False
        assert "__init__" in msg

    def test_missing_next(self):
        code = """
import backtrader as bt

class MyStrategy(bt.Strategy):
    def __init__(self):
        pass
"""
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is False
        assert "next" in msg

    def test_syntax_error(self):
        code = "def foo(:\n    pass"
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is False
        assert "Invalid Python" in msg

    def test_empty_code(self):
        is_valid, msg = validate_backtrader_strategy("")
        assert is_valid is False

    def test_class_not_strategy(self):
        code = """
class MyClass:
    def __init__(self):
        pass
    def next(self):
        pass
"""
        is_valid, msg = validate_backtrader_strategy(code)
        assert is_valid is False
        assert "No valid Backtrader strategy" in msg


class TestExtractClassName:
    """Tests for extract_class_name()."""

    def test_single_class(self):
        code = "class Foo:\n    pass"
        assert extract_class_name(code) == "Foo"

    def test_multiple_classes_returns_first(self):
        code = "class Foo:\n    pass\nclass Bar:\n    pass"
        assert extract_class_name(code) == "Foo"

    def test_no_class(self):
        code = "x = 42"
        assert extract_class_name(code) is None

    def test_syntax_error_returns_none(self):
        code = "def foo(:\n    pass"
        assert extract_class_name(code) is None

    def test_empty_code(self):
        assert extract_class_name("") is None

    def test_bt_strategy(self):
        code = """
import backtrader as bt

class MACrossover(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        pass
"""
        assert extract_class_name(code) == "MACrossover"
