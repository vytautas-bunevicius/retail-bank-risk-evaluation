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

from typing import List

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encodes categorical features using One-Hot and LeaveOneOut encoding.

    This function performs One-Hot encoding for low-cardinality categorical
    features and LeaveOneOut encoding for high-cardinality categorical features.
    It handles the target variable correctly by excluding it from encoding and
    ensures consistent column names between training and testing sets.
    All generated column names are converted to lowercase for consistency.

    Args:
        df_train: The training dataframe.
        df_test: The testing dataframe.
        ohe_features: A list of column names to be One-Hot encoded.
        target_encoded_features: A list of column names to be target encoded.
        target_column: The name of the target variable column.

    Returns:
        A tuple containing the encoded training and testing dataframes.
    """

    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    ohe = OneHotEncoder(cols=ohe_features, use_cat_names=True, handle_unknown='indicator')
    df_train_encoded = ohe.fit_transform(df_train_encoded.drop(columns=[target_column]))
    df_test_encoded = ohe.transform(df_test_encoded)

    df_train_encoded.columns = df_train_encoded.columns.str.lower()
    df_test_encoded.columns = df_test_encoded.columns.str.lower()

    train_cols = set(df_train_encoded.columns)
    test_cols = set(df_test_encoded.columns)

    missing_in_test = list(train_cols - test_cols)
    for column in missing_in_test:
        df_test_encoded[column] = 0

    extra_in_test = list(test_cols - train_cols)
    df_test_encoded = df_test_encoded.drop(columns=extra_in_test, errors='ignore')

    df_test_encoded = df_test_encoded[df_train_encoded.columns]

    for feature in target_encoded_features:
        encoder = LeaveOneOutEncoder(cols=[feature], handle_unknown='value')

        train_encoded_feature = encoder.fit_transform(df_train[feature], df_train[target_column])
        train_encoded_feature.columns = train_encoded_feature.columns.str.lower()
        df_train_encoded[train_encoded_feature.columns] = train_encoded_feature

        test_encoded_feature = encoder.transform(df_test[feature])
        test_encoded_feature.columns = test_encoded_feature.columns.str.lower()
        df_test_encoded[test_encoded_feature.columns] = test_encoded_feature


    df_train_encoded = df_train_encoded.drop(columns=target_encoded_features)
    df_test_encoded = df_test_encoded.drop(columns=target_encoded_features)

    df_train_encoded[target_column] = df_train[target_column]

    return df_train_encoded, df_test_encoded
