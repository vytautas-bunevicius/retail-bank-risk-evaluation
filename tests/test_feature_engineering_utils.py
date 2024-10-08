"""
Test module for retail_bank_risk feature engineering utilities.

This module contains unit tests for various feature engineering and encoding
functions used in the retail bank risk assessment project. It uses pytest
for test organization and execution.

The module tests the following main functionalities:
1. Age and numeric binning
2. Derived feature creation (e.g., ratios, flags)
3. Categorical feature encoding (binary, one-hot, target, ordinal)
4. Column name cleaning
5. Handling of constant and unknown categories
6. Integration of all feature engineering steps

Each test function corresponds to a specific utility function from the
retail_bank_risk.feature_engineering_utils module, ensuring that these
functions work as expected with sample data.

Fixtures:
- sample_df: Provides a sample Pandas DataFrame for testing

The tests cover various scenarios, including:
- Normal operation with valid inputs
- Error handling for missing or invalid inputs
- Edge cases such as constant columns and unknown categories
- Integration of multiple feature engineering steps

Note: Some tests check for proper error handling by asserting that specific
exceptions are raised under certain conditions.
"""

from typing import List

import pytest
import pandas as pd
import numpy as np

from retail_bank_risk.feature_engineering_utils import (
    bin_age_into_groups,
    bin_numeric_into_quantiles,
    create_binned_features,
    calculate_ratio,
    create_derived_features,
    engineer_features,
    encode_categorical_features,
    clean_column_names,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a sample DataFrame for testing."""
    data = {
        "days_birth": [10000, 20000, 30000, 40000, 50000],
        "amt_income_total": [50000, 60000, 70000, 80000, 90000],
        "amt_credit": [20000, 25000, 30000, 35000, 40000],
        "amt_goods_price": [15000, 20000, 25000, 30000, 35000],
        "amt_annuity": [5000, 6000, 7000, 8000, 9000],
        "ext_source_2": [0.5, 0.6, 0.7, 0.8, 0.9],
        "ext_source_3": [0.4, 0.5, 0.6, 0.7, 0.8],
    }
    return pd.DataFrame(data)


def test_bin_age_into_groups(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test bin_age_into_groups with valid age data."""
    binned = bin_age_into_groups(sample_df, "days_birth")
    expected_labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]
    assert binned.dtype.name == "category"
    assert all(label in binned.cat.categories for label in expected_labels)


def test_bin_age_into_groups_missing_column() -> None:
    """Test bin_age_into_groups raises KeyError when age_column is missing."""
    df = pd.DataFrame({"other_column": [1, 2, 3]})
    with pytest.raises(KeyError):
        bin_age_into_groups(df, "days_birth")


def test_bin_numeric_into_quantiles(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test bin_numeric_into_quantiles with valid numeric data."""
    binned = bin_numeric_into_quantiles(sample_df, "amt_income_total", 3)
    assert binned.dtype.name == "category"
    assert binned.cat.categories.tolist() == ["Q1", "Q2", "Q3"]


def test_bin_numeric_into_quantiles_single_unique(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test bin_numeric_into_quantiles with a column having a single unique value."""
    df = sample_df.copy()
    df["constant"] = 100
    binned = bin_numeric_into_quantiles(df, "constant")
    expected = pd.Series(["Q1"] * len(df), index=df.index)
    pd.testing.assert_series_equal(binned, expected)


def test_create_binned_features(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test create_binned_features adds the correct binned columns."""
    binned_df = create_binned_features(sample_df)
    assert "age_group" in binned_df.columns
    assert "income_group" in binned_df.columns
    assert "credit_amount_group" in binned_df.columns
    assert binned_df["age_group"].dtype.name == "category"
    assert binned_df["income_group"].dtype.name == "category"
    assert binned_df["credit_amount_group"].dtype.name == "category"


def test_create_binned_features_missing_columns() -> None:
    """Test create_binned_features raises
        KeyError when required columns are missing."""
    df = pd.DataFrame({"amt_income_total": [50000, 60000]})
    with pytest.raises(KeyError):
        create_binned_features(df)


def test_calculate_ratio(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test calculate_ratio with valid numerator and denominator."""
    ratio = calculate_ratio(sample_df, "amt_credit", "amt_income_total")
    expected = sample_df["amt_credit"] / sample_df["amt_income_total"]
    np.testing.assert_array_almost_equal(ratio, expected)


def test_calculate_ratio_with_zero_denominator(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test calculate_ratio handles division by zero correctly."""
    df = sample_df.copy()
    df.loc[2, "amt_income_total"] = 0
    ratio = calculate_ratio(
        df, "amt_credit", "amt_income_total", fill_value=999999
    )

    assert ratio[2] == 999999
    assert not np.isinf(ratio).any()


def test_create_derived_features(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test create_derived_features adds the correct derived columns."""
    derived_df = create_derived_features(sample_df)
    assert "zero_income_flag" in derived_df.columns
    assert "debt_to_income_ratio" in derived_df.columns
    assert "credit_to_goods_ratio" in derived_df.columns
    assert "annuity_to_income_ratio" in derived_df.columns
    assert "credit_exceeds_goods" in derived_df.columns
    assert derived_df["zero_income_flag"].dtype == int
    assert derived_df["credit_exceeds_goods"].dtype == int


def test_create_derived_features_missing_columns() -> None:
    """Test create_derived_features raises KeyError when required columns are missing."""
    df = pd.DataFrame({"amt_credit": [20000, 25000]})
    with pytest.raises(KeyError):
        create_derived_features(df)


def test_engineer_features(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Test engineer_features integrates binned and derived features correctly."""
    engineered_df = engineer_features(sample_df)
    expected_columns = [
        "days_birth",
        "amt_income_total",
        "amt_credit",
        "amt_goods_price",
        "amt_annuity",
        "ext_source_2",
        "ext_source_3",
        "age_group",
        "income_group",
        "credit_amount_group",
        "zero_income_flag",
        "debt_to_income_ratio",
        "credit_to_goods_ratio",
        "annuity_to_income_ratio",
        "credit_exceeds_goods",
    ]
    for col in expected_columns:
        assert col in engineered_df.columns


def test_encode_categorical_features_binary(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features with binary features."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["binary_feat"] = ["yes", "no", "yes", "no", "yes"]
    df_test["binary_feat"] = ["no", "no", "yes", "yes", "no"]
    binary_features = ["binary_feat"]
    categorical_features = []
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    assert "binary_feat" in encoded_train.columns
    assert "binary_feat" in encoded_test.columns
    assert encoded_train["binary_feat"].dtype == int
    assert encoded_test["binary_feat"].dtype == int
    assert set(encoded_train["binary_feat"].unique()) == {0, 1}


def test_encode_categorical_features_one_hot(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features with categorical features."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["cat_feat"] = ["A", "B", "A", "C", "B"]
    df_test["cat_feat"] = ["B", "C", "A", "A", "B"]
    binary_features: List[str] = []
    categorical_features = ["cat_feat"]
    encoded_train, _ = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )

    generated_columns = encoded_train.columns
    expected_columns = [
        col for col in generated_columns if col.startswith("cat_feat")
    ]

    assert len(expected_columns) > 0


def test_encode_categorical_features_target_encoding(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features with target encoding."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["target"] = [0, 1, 0, 1, 0]
    df_train["target_feat"] = ["low", "high", "medium", "high", "low"]
    df_test["target_feat"] = ["medium", "low", "high", "medium", "high"]
    binary_features: List[str] = []
    categorical_features: List[str] = []
    target_encoded_features = ["target_feat"]
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
        target_encoded_features=target_encoded_features,
        target_column="target",
    )
    assert "target_feat" in encoded_train.columns
    assert "target_feat" in encoded_test.columns
    assert encoded_train["target_feat"].dtype == float
    assert encoded_test["target_feat"].dtype == float


def test_encode_categorical_features_ordinal_encoding(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features with ordinal features."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["ordinal_feat"] = ["low", "medium", "high", "medium", "low"]
    df_test["ordinal_feat"] = ["high", "low", "medium", "high", "medium"]
    binary_features: List[str] = []
    categorical_features: List[str] = []
    ordinal_features = ["ordinal_feat"]
    ordinal_orders = {"ordinal_feat": ["low", "medium", "high"]}
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        ordinal_orders=ordinal_orders,
    )
    assert "ordinal_feat" in encoded_train.columns
    assert "ordinal_feat" in encoded_test.columns
    assert encoded_train["ordinal_feat"].dtype == float
    assert encoded_test["ordinal_feat"].dtype == float
    assert encoded_train["ordinal_feat"].min() == 0
    assert encoded_train["ordinal_feat"].max() == 2


def test_encode_categorical_features_missing_target_column(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features raises ValueError when target_column is missing."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    target_encoded_features = ["ext_source_2"]
    with pytest.raises(ValueError):
        encode_categorical_features(
            df_train=df_train,
            df_test=df_test,
            binary_features=[],
            categorical_features=[],
            target_encoded_features=target_encoded_features,
        )


def test_encode_categorical_features_missing_ordinal_orders(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features raises ValueError when ordinal_orders missing."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["ordinal_feat"] = ["low", "medium", "high", "medium", "low"]
    df_test["ordinal_feat"] = ["high", "low", "medium", "high", "medium"]
    ordinal_features = ["ordinal_feat"]
    with pytest.raises(ValueError):
        encode_categorical_features(
            df_train=df_train,
            df_test=df_test,
            binary_features=[],
            categorical_features=[],
            ordinal_features=ordinal_features,
        )


def test_encode_categorical_features_constant_columns(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features removes constant columns."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["constant_feat"] = "constant"
    df_test["constant_feat"] = "constant"
    binary_features: List[str] = []
    categorical_features: List[str] = ["constant_feat"]
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    assert "constant_feat" not in encoded_train.columns
    assert "constant_feat" not in encoded_test.columns


def test_encode_categorical_features_placeholders(
    sample_df: pd.DataFrame,  # pylint: disable=W0621
) -> None:
    """Test encode_categorical_features handles placeholder values correctly."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["binary_feat"] = ["yes", "xna", "no", "yes", "xna"]
    df_test["binary_feat"] = ["no", "xna", "yes", "xna", "no"]
    binary_features = ["binary_feat"]
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=[],
    )

    assert set(encoded_train["binary_feat"].unique()) == {0, 1, 2}
    assert set(encoded_test["binary_feat"].unique()) == {0, 1, 2}


def test_clean_column_names() -> None:
    """Test that column names are cleaned correctly."""

    df = pd.DataFrame(
        {
            "Column 1": [1, 2],
            "Column-2": [3, 4],
            "Column@3": [5, 6],
        }
    )
    cleaned_df = clean_column_names(df)
    expected_columns = ["column_1", "column_2", "column_3"]
    assert list(cleaned_df.columns) == expected_columns


def test_encode_categorical_features_unknown_categories(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features handles unknown categories in test data."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["cat_feat"] = ["A", "B", "A", "C", "B"]
    df_test["cat_feat"] = ["B", "C", "D", "A", "B"]
    binary_features: List[str] = []
    categorical_features = ["cat_feat"]
    encoded_train, _ = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )

    generated_columns = encoded_train.columns
    expected_columns = [
        col for col in generated_columns if col.startswith("cat_feat")
    ]

    assert len(expected_columns) > 0
    assert "cat_feat_D" not in encoded_train.columns


def test_encode_categorical_features_reindexing(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features reindexes test DataFrame correctly."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["cat_feat"] = ["A", "B", "A", "C", "B"]
    df_test["cat_feat"] = ["B", "C", "A", "A", "B"]
    binary_features: List[str] = []
    categorical_features = ["cat_feat"]
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    assert list(encoded_test.columns) == list(encoded_train.columns)


def test_encode_categorical_features_target_column_in_test(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features drops target_column from test DataFrame."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["target"] = [0, 1, 0, 1, 0]
    df_test["target"] = [1, 0, 1, 0, 1]
    binary_features: List[str] = []
    categorical_features: List[str] = []
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
        target_column="target",
    )
    assert "target" in encoded_train.columns
    assert "target" not in encoded_test.columns
    assert "target" in encoded_train.columns


def test_encode_categorical_features_constant_columns_after_encoding(
    sample_df: pd.DataFrame,  #pylint: disable=W0621
) -> None:
    """Test encode_categorical_features removes constant columns post-encoding."""
    df_train = sample_df.copy()
    df_test = sample_df.copy()
    df_train["constant_cat"] = "constant"
    df_test["constant_cat"] = "constant"
    binary_features: List[str] = []
    categorical_features = ["constant_cat"]
    encoded_train, encoded_test = encode_categorical_features(
        df_train=df_train,
        df_test=df_test,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    assert "constant_cat" not in encoded_train.columns
    assert "constant_cat" not in encoded_test.columns


def test_engineer_features_integration(
    sample_df: pd.DataFrame, #pylint: disable=W0621
) -> None:
    """Integration test for engineer_features combining all feature engineering steps."""
    engineered_df = engineer_features(sample_df)
    assert "age_group" in engineered_df.columns
    assert "income_group" in engineered_df.columns
    assert "credit_amount_group" in engineered_df.columns
    assert "zero_income_flag" in engineered_df.columns
    assert "debt_to_income_ratio" in engineered_df.columns
    assert "credit_to_goods_ratio" in engineered_df.columns
    assert "annuity_to_income_ratio" in engineered_df.columns
    assert "credit_exceeds_goods" in engineered_df.columns
    original_columns = [
        "days_birth",
        "amt_income_total",
        "amt_credit",
        "amt_goods_price",
        "amt_annuity",
        "ext_source_2",
        "ext_source_3",
    ]
    for col in original_columns:
        assert col in engineered_df.columns
