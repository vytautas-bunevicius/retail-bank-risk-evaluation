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

import numpy as np
import pandas as pd


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
    df["employed_to_age_ratio"] = calculate_ratio(
        df,
        "days_employed",
        "days_birth",
        "employed_to_age_ratio",
        fill_value=1.0,
    )
    df["ext_source_mean"] = df[
        ["ext_source_1", "ext_source_2", "ext_source_3"]
    ].mean(axis=1)
    df["credit_exceeds_goods"] = (
        df["amt_credit"] > df["amt_goods_price"]
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the input DataFrame.

    This function creates both binned versions of continuous variables
    and new derived features based on domain knowledge.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with all engineered features added.
    """
    df = create_binned_features(df)
    df = create_derived_features(df)
    return df
