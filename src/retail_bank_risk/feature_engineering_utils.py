"""
This module provides functions for engineering features from a given DataFrame.
It includes methods for binning continuous variables into categorical groups,
calculating ratio-based features, and creating derived features based on domain knowledge.

Functions:
    bin_age_into_groups(df: pd.DataFrame, age_column: str) -> pd.Series
    bin_numeric_into_quantiles(df: pd.DataFrame, column: str, num_bins: int = 5) -> pd.Series
    create_binned_features(df: pd.DataFrame) -> pd.DataFrame
    calculate_ratio(df: pd.DataFrame, numerator: str,
            denominator: str, _: str, fill_value: float = 0.0) -> pd.Series
    create_derived_features(df: pd.DataFrame) -> pd.DataFrame
    engineer_features(df: pd.DataFrame) -> pd.DataFrame
"""

from typing import List, Tuple

import pandas as pd
import numpy as np
from category_encoders import LeaveOneOutEncoder, OneHotEncoder


def bin_age_into_groups(df: pd.DataFrame, age_column: str) -> pd.Series:
    """
    Bin age values into predefined groups.

    Args:
        df (pd.DataFrame): The input DataFrame.
        age_column (str): The name of the column containing age values in days.

    Returns:
        pd.Series: A series with binned age groups.
    """
    return pd.cut(
        -df[age_column] / 365,
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )


def bin_numeric_into_quantiles(
    df: pd.DataFrame, column: str, num_bins: int = 5
) -> pd.Series:
    """
    Bin numeric values into quantile-based groups.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to bin.
        num_bins (int): The number of bins to create. Defaults to 5.

    Returns:
        pd.Series: A series with binned values.
    """
    return pd.qcut(
        df[column], q=num_bins, labels=[f"Q{i+1}" for i in range(num_bins)]
    )


def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binned versions of continuous variables.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with additional binned features.
    """
    df = df.copy()
    df["age_group"] = bin_age_into_groups(df, "days_birth")
    df["income_group"] = bin_numeric_into_quantiles(df, "amt_income_total")
    df["credit_amount_group"] = bin_numeric_into_quantiles(df, "amt_credit")
    return df


def calculate_ratio(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    _: str,
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Calculate the ratio between two columns, handling division by zero.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerator (str): The name of the column to use as numerator.
        denominator (str): The name of the column to use as denominator.
        _: str: Placeholder for the new column name (unused).
        fill_value (float): The value to use when denominator is zero. Defaults to 0.0.

    Returns:
        pd.Series: A series with the calculated ratios.
    """
    return np.where(
        df[denominator] != 0, df[numerator] / df[denominator], fill_value
    )


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features derived from existing ones based on domain knowledge.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with additional derived features.
    """
    df = df.copy()
    df["debt_to_income_ratio"] = calculate_ratio(
        df, "amt_credit", "amt_income_total", "debt_to_income_ratio"
    )
    df["credit_to_goods_ratio"] = calculate_ratio(
        df, "amt_credit", "amt_goods_price", "credit_to_goods_ratio"
    )
    df["annuity_to_income_ratio"] = calculate_ratio(
        df, "amt_annuity", "amt_income_total", "annuity_to_income_ratio"
    )

    df["ext_source_mean"] = df[["ext_source_2", "ext_source_3"]].mean(axis=1)
    df["credit_exceeds_goods"] = (
        df["amt_credit"] > df["amt_goods_price"]
    ).astype(int)
    return df


def encode_categorical_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ohe_features: List[str],
    target_encoded_features: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes categorical features using One-Hot Encoding and Leave-One-Out Encoding.

    This function performs One-Hot Encoding for low-cardinality categorical
    features and Leave-One-Out Encoding for high-cardinality categorical features.
    It excludes target-encoded features from One-Hot Encoding to prevent redundant processing.
    The function replaces specific placeholder values like 'xna' with NaN
            to correctly handle missing values.
    It ensures consistent column names between training and testing sets by aligning them.
    All generated column names are converted to lowercase for consistency.
    Unwanted encoded columns containing '_xna' or '-1' are dropped immediately after encoding.
    Constant columns with only one unique value are also removed.

    Args:
        df_train (pd.DataFrame): The training dataframe.
        df_test (pd.DataFrame): The testing dataframe.
        ohe_features (List[str]): List of column names to be One-Hot encoded.
        target_encoded_features (List[str]): List of column names to be Leave-One-Out encoded.
        target_column (str): The name of the target variable column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the encoded
                training and testing dataframes.
    """

    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    categorical_features = ohe_features + target_encoded_features

    placeholders = ["xna"]

    for feature in categorical_features:
        if feature in df_train_encoded.columns:
            df_train_encoded[feature] = df_train_encoded[feature].replace(
                placeholders, np.nan
            )
        if feature in df_test_encoded.columns:
            df_test_encoded[feature] = df_test_encoded[feature].replace(
                placeholders, np.nan
            )

    ohe_features_exclusive = [
        feat for feat in ohe_features if feat not in target_encoded_features
    ]

    ohe = OneHotEncoder(
        cols=ohe_features_exclusive, use_cat_names=True, handle_unknown="ignore"
    )

    if target_column in df_train_encoded.columns:
        df_train_ohe = df_train_encoded.drop(columns=[target_column])
    else:
        df_train_ohe = df_train_encoded.copy()

    df_train_encoded = ohe.fit_transform(df_train_ohe)

    if target_column in df_test_encoded.columns:
        df_test_ohe = df_test_encoded.drop(columns=[target_column])
    else:
        df_test_ohe = df_test_encoded.copy()

    df_test_encoded = ohe.transform(df_test_ohe)

    df_train_encoded.columns = df_train_encoded.columns.str.lower()
    df_test_encoded.columns = df_test_encoded.columns.str.lower()

    df_test_encoded = df_test_encoded.reindex(
        columns=df_train_encoded.columns, fill_value=0
    )

    unwanted_suffixes = ["_xna", "-1"]
    unwanted_cols = [
        col
        for col in df_train_encoded.columns
        if any(suffix in col for suffix in unwanted_suffixes)
    ]
    if unwanted_cols:
        df_train_encoded = df_train_encoded.drop(columns=unwanted_cols)
        df_test_encoded = df_test_encoded.drop(
            columns=unwanted_cols, errors="ignore"
        )

    loo = LeaveOneOutEncoder(
        cols=target_encoded_features, handle_unknown="value", sigma=0.0
    )

    df_train_encoded[target_encoded_features] = loo.fit_transform(
        df_train[target_encoded_features], df_train[target_column]
    )

    df_test_encoded[target_encoded_features] = loo.transform(
        df_test[target_encoded_features]
    )

    if target_column in df_train.columns:
        df_train_encoded[target_column] = df_train[target_column].values

    # Remove constant columns
    constant_cols = [
        col
        for col in df_train_encoded.columns
        if df_train_encoded[col].nunique() == 1
    ]
    if constant_cols:
        df_train_encoded = df_train_encoded.drop(columns=constant_cols)
        df_test_encoded = df_test_encoded.drop(
            columns=constant_cols, errors="ignore"
        )

    return df_train_encoded, df_test_encoded
