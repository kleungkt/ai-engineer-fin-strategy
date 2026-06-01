"""Tests for the optimizer.parameter_optimizer module."""

import pytest
import pandas as pd
import numpy as np
from optimizer.parameter_optimizer import (
    OptimizationResult,
    grid_search,
    random_search,
    _extract_metric,
    _generate_param_combinations,
    _sample_random_params,
)


@pytest.fixture
def sample_data():
    """Generate a small DataFrame for optimizer tests."""
    np.random.seed(0)
    dates = pd.bdate_range("2024-01-01", periods=100)
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.02, 100))
    data = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, 100),
    })
    return data


def mock_strategy_fn(data, period=20, threshold=0.5):
    """Mock strategy function that returns synthetic metrics."""
    # Simple deterministic scoring based on params
    score = period * 0.01 + threshold * 0.1
    return {
        "sharpe_ratio": score,
        "total_return": score * 0.5,
        "annual_return": score * 0.3,
        "max_drawdown": 0.1,
        "win_rate": 0.55,
        "sortino_ratio": score * 1.2,
        "calmar_ratio": score * 0.8,
    }


class TestOptimizationResult:
    """Tests for OptimizationResult model."""

    def test_structure(self):
        result = OptimizationResult(
            best_params={"period": 20},
            best_score=1.5,
            all_results=[{"params": {"period": 20}, "score": 1.5}],
            method="grid_search",
        )
        assert result.best_params == {"period": 20}
        assert result.best_score == 1.5
        assert result.method == "grid_search"
        assert len(result.all_results) == 1

    def test_default_all_results(self):
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            method="grid_search",
        )
        assert result.all_results == []

    def test_is_pydantic_model(self):
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            method="grid_search",
        )
        d = result.model_dump()
        assert "best_params" in d
        assert "best_score" in d
        assert "all_results" in d
        assert "method" in d


class TestExtractMetric:
    """Tests for _extract_metric()."""

    def test_sharpe(self):
        results = {"sharpe_ratio": 1.5}
        assert _extract_metric(results, "sharpe") == 1.5

    def test_total_return(self):
        results = {"total_return": 0.25}
        assert _extract_metric(results, "return") == 0.25

    def test_win_rate(self):
        results = {"win_rate": 0.65}
        assert _extract_metric(results, "win_rate") == 0.65

    def test_missing_metric_returns_zero(self):
        results = {"sharpe_ratio": 1.5}
        assert _extract_metric(results, "win_rate") == 0.0

    def test_none_value_returns_zero(self):
        results = {"sharpe_ratio": None}
        assert _extract_metric(results, "sharpe") == 0.0

    def test_alias_resolution(self):
        results = {"total_return": 0.15}
        assert _extract_metric(results, "total_return") == 0.15


class TestGenerateParamCombinations:
    """Tests for _generate_param_combinations()."""

    def test_single_param(self):
        grid = {"a": [1, 2, 3]}
        combos = _generate_param_combinations(grid)
        assert len(combos) == 3
        assert {"a": 1} in combos
        assert {"a": 3} in combos

    def test_two_params(self):
        grid = {"a": [1, 2], "b": [10, 20]}
        combos = _generate_param_combinations(grid)
        assert len(combos) == 4
        assert {"a": 1, "b": 10} in combos
        assert {"a": 2, "b": 20} in combos

    def test_empty_grid(self):
        grid = {}
        combos = _generate_param_combinations(grid)
        # product of empty iterable yields single empty tuple
        assert len(combos) == 1
        assert combos[0] == {}


class TestGridSearch:
    """Tests for grid_search()."""

    def test_basic_grid_search(self, sample_data):
        param_grid = {"period": [10, 20], "threshold": [0.3, 0.7]}
        result = grid_search(sample_data, mock_strategy_fn, param_grid, metric="sharpe")

        assert isinstance(result, OptimizationResult)
        assert result.method == "grid_search"
        assert len(result.all_results) == 4
        # Best should be period=20, threshold=0.7 (highest score)
        assert result.best_params == {"period": 20, "threshold": 0.7}

    def test_best_score_is_correct(self, sample_data):
        param_grid = {"period": [10, 20], "threshold": [0.3, 0.7]}
        result = grid_search(sample_data, mock_strategy_fn, param_grid, metric="sharpe")
        # max score = 20 * 0.01 + 0.7 * 0.1 = 0.27
        assert abs(result.best_score - 0.27) < 1e-6

    def test_single_param(self, sample_data):
        param_grid = {"period": [5]}
        result = grid_search(sample_data, mock_strategy_fn, param_grid, metric="sharpe")
        assert len(result.all_results) == 1
        assert result.best_params == {"period": 5}

    def test_all_results_have_score(self, sample_data):
        param_grid = {"period": [10, 20]}
        result = grid_search(sample_data, mock_strategy_fn, param_grid, metric="sharpe")
        for r in result.all_results:
            assert "params" in r
            assert "score" in r

    def test_with_failing_strategy(self, sample_data):
        """Grid search should handle strategy failures gracefully."""
        def bad_strategy(data, period=10):
            if period == 99:
                raise ValueError("bad params")
            return {"sharpe_ratio": 1.0}

        param_grid = {"period": [10, 99]}
        result = grid_search(sample_data, bad_strategy, param_grid, metric="sharpe")
        assert len(result.all_results) == 2
        # The failing run should have score 0.0
        failing = [r for r in result.all_results if r["params"]["period"] == 99]
        assert len(failing) == 1
        assert failing[0]["score"] == 0.0
        assert "error" in failing[0]


class TestRandomSearch:
    """Tests for random_search()."""

    def test_basic_random_search(self, sample_data):
        param_ranges = {"period": [5, 50], "threshold": [0.1, 0.9]}
        result = random_search(
            sample_data, mock_strategy_fn, param_ranges, n_trials=20, metric="sharpe"
        )

        assert isinstance(result, OptimizationResult)
        assert result.method == "random_search"
        assert len(result.all_results) == 20
        assert result.best_score > float("-inf")

    def test_best_params_in_range(self, sample_data):
        param_ranges = {"period": [5, 50], "threshold": [0.1, 0.9]}
        result = random_search(
            sample_data, mock_strategy_fn, param_ranges, n_trials=50, metric="sharpe"
        )
        # period should be in [5, 50] and threshold in [0.1, 0.9]
        assert 5 <= result.best_params["period"] <= 50
        assert 0.1 <= result.best_params["threshold"] <= 0.9

    def test_with_choice_values(self, sample_data):
        param_ranges = {"period": {"type": "int", "low": 5, "high": 20}}
        result = random_search(
            sample_data, mock_strategy_fn, param_ranges, n_trials=10, metric="sharpe"
        )
        assert len(result.all_results) == 10


class TestSampleRandomParams:
    """Tests for _sample_random_params()."""

    def test_list_range(self):
        params = _sample_random_params({"x": [10, 100]})
        assert 10 <= params["x"] <= 100

    def test_int_range(self):
        params = _sample_random_params({"x": (1, 10)})
        assert isinstance(params["x"], int)
        assert 1 <= params["x"] <= 10

    def test_float_range(self):
        params = _sample_random_params({"x": (0.1, 0.9)})
        assert isinstance(params["x"], float)
        assert 0.1 <= params["x"] <= 0.9

    def test_fixed_value(self):
        params = _sample_random_params({"x": 42})
        assert params["x"] == 42

    def test_dict_type_choice(self):
        params = _sample_random_params({"x": {"type": "choice", "values": [1, 2, 3]}})
        assert params["x"] in [1, 2, 3]

    def test_dict_type_int(self):
        params = _sample_random_params({"x": {"type": "int", "low": 1, "high": 10}})
        assert isinstance(params["x"], int)
        assert 1 <= params["x"] <= 10
