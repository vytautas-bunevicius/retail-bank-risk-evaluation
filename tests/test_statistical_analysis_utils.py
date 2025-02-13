"""
Test module for retail_bank_risk statistical analysis utilities.

This module contains unit tests for various statistical analysis functions
used in the retail bank risk assessment project. It uses pytest for test
organization and execution.

The module tests the following main functionalities:
1. Cliff's Delta calculation (vectorized and bootstrapped)
2. Cramer's V calculation (bootstrapped)
3. P-value simulation
4. Feature analysis (numerical and categorical)
5. Statistical analysis execution
6. Target variable diagnostics

Each test function corresponds to a specific utility function from the
retail_bank_risk.statistical_analysis_utils module, ensuring that these
functions work as expected with sample data.

Fixtures:
- fx_sample_data_group_one/two: Sample data for two groups
- fx_sample_data_numerical: Sample numerical data
- fx_empty_array: Empty numpy array
- fx_sample_contingency_table: Sample contingency table
- fx_empty_contingency_table: Empty contingency table
- fx_sample_dataframe_numerical: Sample DataFrame with numerical and categorical features
- fx_sample_dataframe_multiple: Sample DataFrame with multiple features

The tests cover various scenarios, including:
- Normal operation with valid inputs
- Edge cases (e.g., empty data, single iteration)
- Error handling for invalid inputs
- Behavior with different data types (numerical, categorical, boolean)
- Multiple feature analysis
- Target variable diagnostics

Note: Some tests use pytest.approx for floating-point comparisons to
account for small numerical differences.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from retail_bank_risk.statistical_analysis_utils import (
    analyze_feature,
    bootstrap_cliff_delta,
    bootstrap_cramers_v,
    diagnose_target,
    run_statistical_analysis,
    simulate_p_value,
    vectorized_cliff_delta,
)


@pytest.fixture
def fx_sample_data_group_one() -> np.ndarray:
    """Fixture that provides sample data for group one."""
    return np.array([3, 4, 5])


@pytest.fixture
def fx_sample_data_group_two() -> np.ndarray:
    """Fixture that provides sample data for group two."""
    return np.array([1, 2])


@pytest.fixture
def fx_sample_data_numerical() -> np.ndarray:
    """Fixture that provides sample numerical data."""
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def fx_empty_array() -> np.ndarray:
    """Fixture that provides an empty numpy array."""
    return np.array([])


@pytest.fixture
def fx_sample_contingency_table() -> pd.DataFrame:
    """Fixture that provides a sample contingency table as a pandas DataFrame."""
    return pd.DataFrame(
        [[10, 20], [30, 40]], columns=["A", "B"], index=["X", "Y"]
    )


@pytest.fixture
def fx_empty_contingency_table() -> pd.DataFrame:
    """Fixture that provides an empty contingency table as a pandas DataFrame."""
    return pd.DataFrame([[0, 0], [0, 0]])


@pytest.fixture
def fx_sample_dataframe_numerical() -> pd.DataFrame:
    """Fixture that provides a sample numerical and categorical feature DataFrame."""
    return pd.DataFrame(
        {
            "numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "categorical_feature": [
                "A",
                "B",
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "C",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def fx_sample_dataframe_multiple() -> pd.DataFrame:
    """Fixture that provides a sample DataFrame with multiple numerical and
    categorical features."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "numeric_feature1": np.random.normal(0, 1, 100),
            "numeric_feature2": np.random.normal(5, 2, 100),
            "categorical_feature1": np.random.choice(["A", "B", "C"], 100),
            "categorical_feature2": np.random.choice(["X", "Y"], 100),
            "target": np.random.choice([0, 1], 100),
        }
    )


def test_vectorized_cliff_delta_positive(
    fx_sample_data_group_one: np.ndarray, #pylint: disable=W0621
    fx_sample_data_group_two: np.ndarray #pylint: disable=W0621
) -> None:
    """Test vectorized Cliff's delta function with positive result."""
    result = vectorized_cliff_delta(
        fx_sample_data_group_one, fx_sample_data_group_two
    )
    assert result == 1.0


def test_vectorized_cliff_delta_negative(
    fx_sample_data_group_two: np.ndarray, #pylint: disable=W0621
    fx_sample_data_group_one: np.ndarray #pylint: disable=W0621
) -> None:
    """Test vectorized Cliff's delta function with negative result."""
    result = vectorized_cliff_delta(
        fx_sample_data_group_two, fx_sample_data_group_one
    )
    assert result == -1.0


def test_vectorized_cliff_delta_equality(fx_sample_data_group_one: np.ndarray) -> None: #pylint: disable=W0621
    """Test vectorized Cliff's delta function with equal groups."""
    result = vectorized_cliff_delta(
        fx_sample_data_group_one, fx_sample_data_group_one
    )
    assert result == 0.0


def test_vectorized_cliff_delta_mixed() -> None:
    """Test vectorized Cliff's delta function with mixed group data."""
    group_one: np.ndarray = np.array([1, 3, 5])
    group_two: np.ndarray = np.array([2, 4, 6])
    result: float = vectorized_cliff_delta(group_one, group_two)
    assert result == pytest.approx(0.0, abs=0.5)


def test_vectorized_cliff_delta_empty_group(
    fx_empty_array: np.ndarray, #pylint: disable=W0621
    fx_sample_data_group_two: np.ndarray #pylint: disable=W0621
) -> None:
    """Test vectorized Cliff's delta function with an empty group."""
    result: float = vectorized_cliff_delta(fx_empty_array, fx_sample_data_group_two)
    assert np.isnan(result)


def test_bootstrap_cliff_delta_confidence_interval(
    fx_sample_data_numerical: np.ndarray #pylint: disable=W0621
) -> None:
    """Test bootstrap Cliff's delta confidence interval calculation."""
    lower: float
    upper: float
    lower, upper = bootstrap_cliff_delta(
        fx_sample_data_numerical, fx_sample_data_numerical, num_iterations=1000
    )
    assert -1.0 <= lower <= 1.0
    assert -1.0 <= upper <= 1.0


def test_bootstrap_cliff_delta_single_iteration(
    fx_sample_data_group_one: np.ndarray, #pylint: disable=W0621
    fx_sample_data_group_two: np.ndarray #pylint: disable=W0621
) -> None:
    """Test bootstrap Cliff's delta function with a single iteration."""
    lower: float
    upper: float
    lower, upper = bootstrap_cliff_delta(
        fx_sample_data_group_one, fx_sample_data_group_two, num_iterations=1
    )
    expected: float = vectorized_cliff_delta(
        fx_sample_data_group_one, fx_sample_data_group_two
    )
    assert lower == expected
    assert upper == expected


def test_bootstrap_cliff_delta_empty_data(
    fx_empty_array: np.ndarray, #pylint: disable=W0621
    fx_sample_data_group_two: np.ndarray #pylint: disable=W0621
) -> None:
    """Test bootstrap Cliff's delta function with empty data."""
    result: Tuple[float, float] = bootstrap_cliff_delta(
        fx_empty_array, fx_sample_data_group_two, num_iterations=100
    )
    assert np.isnan(result[0]) and np.isnan(result[1])


def test_bootstrap_cramers_v_confidence_interval(
    fx_sample_contingency_table: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test bootstrap Cramer's V confidence interval calculation."""
    lower: float
    upper: float
    lower, upper = bootstrap_cramers_v(
        fx_sample_contingency_table, num_iterations=1000
    )
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0


def test_bootstrap_cramers_v_empty_table(
    fx_empty_contingency_table: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test bootstrap Cramer's V function with an empty contingency table."""
    with pytest.raises(ValueError):
        bootstrap_cramers_v(fx_empty_contingency_table, num_iterations=100)


def test_simulate_p_value_valid(
    fx_sample_contingency_table: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test p-value simulation with a valid contingency table."""
    p_value: float = simulate_p_value(
        fx_sample_contingency_table, num_simulations=1000
    )
    assert 0.0 <= p_value <= 1.0


def test_simulate_p_value_zero_observed() -> None:
    """Test p-value simulation with a contingency table with equal observed values."""
    contingency_table: pd.DataFrame = pd.DataFrame(
        [[10, 10], [10, 10]], columns=["A", "B"], index=["X", "Y"]
    )
    p_value: float = simulate_p_value(contingency_table, num_simulations=1000)
    assert p_value == pytest.approx(1.0, 0.01)


def test_simulate_p_value_empty_table(
    fx_empty_contingency_table: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test p-value simulation with an empty contingency table."""
    with pytest.raises(ValueError):
        simulate_p_value(fx_empty_contingency_table, num_simulations=100)


def test_analyze_numerical_feature(
    fx_sample_dataframe_numerical: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test analysis of a numerical feature."""
    feature: str = "numeric_feature"
    _, result = analyze_feature(
        feature, fx_sample_dataframe_numerical, "target"
    )
    result: Optional[dict] = result
    assert result is not None
    assert "p_value" in result
    assert "effect_size" in result
    assert "confidence_interval" in result
    assert result["test"] == "mannwhitneyu"


def test_analyze_categorical_feature(
    fx_sample_dataframe_numerical: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test analysis of a categorical feature."""
    feature: str = "categorical_feature"
    _, result = analyze_feature(
        feature, fx_sample_dataframe_numerical, "target"
    )
    result: Optional[dict] = result
    assert result is not None
    assert "p_value" in result
    assert "effect_size" in result
    assert "confidence_interval" in result
    assert result["test"] in ["chi2", "fisher", "chi2_simulated"]


def test_analyze_nonexistent_feature(
    fx_sample_dataframe_numerical: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test analysis with a nonexistent feature."""
    with pytest.raises(KeyError):
        analyze_feature("nonexistent", fx_sample_dataframe_numerical, "target")


def test_analyze_non_binary_target() -> None:
    """Test analysis with a non-binary target variable."""
    data: pd.DataFrame = pd.DataFrame(
        {
            "numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "categorical_feature": [
                "A",
                "B",
                "A",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "B",
            ],
            "target": [0, 1, 2, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    feature: str = "numeric_feature"
    _, result = analyze_feature(feature, data, "target")
    result: Optional[dict] = result
    assert result is None


def test_analyze_boolean_feature() -> None:
    """Test analysis of a boolean feature."""
    data: pd.DataFrame = pd.DataFrame(
        {
            "boolean_feature": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    feature: str = "boolean_feature"
    _, result = analyze_feature(feature, data, "target")
    result: Optional[dict] = result
    assert result is not None
    assert result["test"] == "mannwhitneyu"


def test_run_statistical_analysis_basic(
    fx_sample_dataframe_multiple: pd.DataFrame #pylint: disable=W0621
) -> None:
    """Test running statistical analysis on multiple features."""
    features: List[str] = [
        "numeric_feature1",
        "numeric_feature2",
        "categorical_feature1",
        "categorical_feature2",
    ]
    result_df: pd.DataFrame = run_statistical_analysis(
        fx_sample_dataframe_multiple, features, "target", auto_correct=True
    )
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 4
    expected_columns: List[str] = [
        "p_value",
        "effect_size",
        "confidence_interval",
        "test",
        "corrected_p_value",
    ]
    assert list(result_df.columns) == expected_columns
    assert len(result_df) > 0  # Ensure there is at least one result


def test_run_statistical_analysis_nonexistent_target(
    fx_sample_dataframe_multiple: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test running statistical analysis with a nonexistent target variable."""
    with pytest.raises(KeyError):
        run_statistical_analysis(
            fx_sample_dataframe_multiple,
            ["numeric_feature1"],
            "nonexistent_target",
        )


def test_diagnose_target(capsys) -> None:
    """Test target variable diagnostics output."""
    data: pd.DataFrame = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    diagnose_target(data, "target")
    captured = capsys.readouterr()
    assert "Target variable diagnostics for 'target':" in captured.out
    assert "Total samples: 10" in captured.out
    assert "Number of unique values: 2" in captured.out
    assert "Value counts:" in captured.out
    assert "Number of NaN values: 0" in captured.out
