"""Tests for the strategy_templates module."""

import pytest
import ast
from strategy_templates import (
    TEMPLATES,
    DEFAULT_PARAMS,
    DESCRIPTIONS,
    list_templates,
    get_template,
    render_template,
)

EXPECTED_TEMPLATES = [
    "ma_crossover",
    "rsi_extreme",
    "macd_signal",
    "bollinger_bounce",
    "momentum_breakout",
    "dual_thrust",
]


class TestListTemplates:
    """Tests for list_templates()."""

    def test_returns_six_templates(self):
        templates = list_templates()
        assert len(templates) == 6

    def test_returns_all_expected_names(self):
        templates = list_templates()
        names = [t["name"] for t in templates]
        for expected in EXPECTED_TEMPLATES:
            assert expected in names

    def test_each_has_name_and_description(self):
        templates = list_templates()
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert isinstance(t["name"], str)
            assert isinstance(t["description"], str)
            assert len(t["description"]) > 0

    def test_sorted_by_name(self):
        templates = list_templates()
        names = [t["name"] for t in templates]
        assert names == sorted(names)


class TestGetTemplate:
    """Tests for get_template()."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATES)
    def test_get_each_template(self, name):
        template = get_template(name)
        assert isinstance(template, str)
        assert len(template) > 0

    def test_invalid_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            get_template("nonexistent_template")

    def test_template_contains_class(self):
        for name in EXPECTED_TEMPLATES:
            template = get_template(name)
            assert "class " in template

    def test_template_contains_backtrader_import(self):
        for name in EXPECTED_TEMPLATES:
            template = get_template(name)
            assert "import backtrader" in template


class TestRenderTemplate:
    """Tests for render_template()."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATES)
    def test_render_with_defaults(self, name):
        """Rendering with default params should produce valid Python."""
        code = render_template(name)
        assert isinstance(code, str)
        # Should be parseable Python
        tree = ast.parse(code)
        assert tree is not None

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATES)
    def test_render_produces_valid_backtrader_strategy(self, name):
        """Rendered code should contain a strategy class with __init__ and next."""
        code = render_template(name)
        assert "bt.Strategy" in code or "backtrader.Strategy" in code
        assert "def __init__" in code
        assert "def next" in code

    def test_render_with_custom_params(self):
        code = render_template("ma_crossover", {"fast_period": 5, "slow_period": 20})
        tree = ast.parse(code)
        assert tree is not None
        # Verify the custom params are in the rendered code
        assert "5" in code
        assert "20" in code

    def test_render_with_override(self):
        default_code = render_template("rsi_extreme")
        custom_code = render_template("rsi_extreme", {"oversold": 25})
        # They should differ because we overrode oversold
        assert default_code != custom_code
        assert "25" in custom_code

    def test_render_invalid_name_raises(self):
        with pytest.raises(KeyError):
            render_template("nonexistent")

    def test_render_with_none_params(self):
        """Passing None should use defaults without error."""
        code = render_template("ma_crossover", params=None)
        tree = ast.parse(code)
        assert tree is not None

    def test_ma_crossover_contains_crossover(self):
        code = render_template("ma_crossover")
        assert "CrossOver" in code

    def test_rsi_extreme_contains_rsi(self):
        code = render_template("rsi_extreme")
        assert "RSI" in code

    def test_bollinger_contains_bollinger(self):
        code = render_template("bollinger_bounce")
        assert "BollingerBands" in code


class TestTemplateDefaults:
    """Tests for DEFAULT_PARAMS consistency."""

    def test_all_templates_have_defaults(self):
        for name in EXPECTED_TEMPLATES:
            assert name in DEFAULT_PARAMS, f"Missing defaults for {name}"

    def test_all_templates_have_descriptions(self):
        for name in EXPECTED_TEMPLATES:
            assert name in DESCRIPTIONS, f"Missing description for {name}"

    def test_defaults_are_dicts(self):
        for name, defaults in DEFAULT_PARAMS.items():
            assert isinstance(defaults, dict)
