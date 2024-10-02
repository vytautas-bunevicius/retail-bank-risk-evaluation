"""
This module provides functions for engineering features from a given DataFrame.
It includes methods for binning continuous variables into categorical groups,
calculating ratio-based features, and creating derived features based on domain
knowledge.

Functions:
    bin_age_into_groups(df: pd.DataFrame, age_column: str) -> pd.Series
    bin_numeric_into_quantiles(df: pd.DataFrame, column: str,
        num_bins: int = 5) -> pd.Series
    create_binned_features(df: pd.DataFrame) -> pd.DataFrame
    calculate_ratio(df: pd.DataFrame, numerator: str, denominator: str,
        fill_value: float = 0.0) -> pd.Series
    create_derived_features(df: pd.DataFrame) -> pd.DataFrame
    engineer_features(df: pd.DataFrame) -> pd.DataFrame
    encode_categorical_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
        binary_features: List[str], categorical_features: List[str],
        target_encoded_features: Optional[List[str]] = None,
        ordinal_features: Optional[List[str]] = None,
        ordinal_orders: Optional[Dict[str, List[str]]] = None,
        target_column: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]
"""
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def bin_age_into_groups(df: pd.DataFrame, age_column: str) -> pd.Series:
    """
    Bin age values into predefined groups.

    Args:
        df: The input DataFrame.
        age_column: The name of the column containing age values in days.

    Returns:
        A Series with binned age groups.
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
        df: The input DataFrame.
        column: The name of the column to bin.
        num_bins: The number of bins to create. Defaults to 5.

    Returns:
        A Series with binned values.
    """
    if df[column].nunique() == 1:
        return pd.Series(["Q1"] * len(df), index=df.index)

    labels = [f"Q{i+1}" for i in range(num_bins)]  # f-string with interpolated variables

    return pd.qcut(df[column], q=num_bins, labels=labels, duplicates="drop")




def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binned versions of continuous variables.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with additional binned features.
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
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Calculate the ratio between two columns, handling division by zero.

    Args:
        df: The input DataFrame.
        numerator: The name of the column to use as numerator.
        denominator: The name of the column to use as denominator.
        fill_value: The value to use when denominator is zero. Defaults to 0.0.

    Returns:
        A Series with the calculated ratios.
    """
    return np.where(
        df[denominator] != 0, df[numerator] / df[denominator], fill_value
    )


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features derived from existing ones based on domain knowledge.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with additional derived features.
    """
    df = df.copy()
    df["debt_to_income_ratio"] = calculate_ratio(
        df, "amt_credit", "amt_income_total"
    )
    df["credit_to_goods_ratio"] = calculate_ratio(
        df, "amt_credit", "amt_goods_price"
    )
    df["annuity_to_income_ratio"] = calculate_ratio(
        df, "amt_annuity", "amt_income_total"
    )
    df["ext_source_mean"] = df[["ext_source_2", "ext_source_3"]].mean(axis=1)
    df["credit_exceeds_goods"] = (
        df["amt_credit"] > df["amt_goods_price"]
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the input DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with engineered features.
    """
    df = create_binned_features(df)
    df = create_derived_features(df)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the column names of a DataFrame.

    This function replaces non-alphanumeric characters (excluding underscores)
    with underscores and converts all column names to lowercase.

    Args:
        df: The DataFrame whose columns are to be cleaned.

    Returns:
        A DataFrame with cleaned column names.
    """
    pattern = re.compile(r"[^A-Za-z0-9_]+")
    new_columns = {}
    for col in df.columns:
        new_col = pattern.sub("_", col)
        new_col = new_col.lower()
        new_columns[col] = new_col
    return df.rename(columns=new_columns)


def encode_categorical_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    binary_features: List[str],
    categorical_features: List[str],
    target_encoded_features: Optional[List[str]] = None,
    ordinal_features: Optional[List[str]] = None,
    ordinal_orders: Optional[Dict[str, List[str]]] = None,
    target_column: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features using various encoding methods.

    This function applies different encoding techniques to categorical features
    in both training and testing DataFrames. It handles binary, categorical,
    target-encoded, and ordinal features.

    Args:
        df_train: The training DataFrame.
        df_test: The testing DataFrame.
        binary_features: List of binary features to be label encoded.
        categorical_features: List of features to be one-hot encoded.
        target_encoded_features: List of features to be leave-one-out encoded.
            Defaults to None.
        ordinal_features: List of ordinal features to encode. Defaults to None.
        ordinal_orders: Mapping of ordinal features to their category orders.
            Defaults to None.
        target_column: The name of the target variable column (present only in
            training data). Defaults to an empty string.

    Returns:
        A tuple containing the encoded training and testing DataFrames.

    Raises:
        ValueError: If target_column is not provided when using
            target_encoded_features, or if ordinal_orders is not provided when
            using ordinal_features.
    """
    if target_encoded_features is None:
        target_encoded_features = []
    if ordinal_features is None:
        ordinal_features = []
    if ordinal_orders is None:
        ordinal_orders = {}

    if target_column and target_column in df_test.columns:
        df_test = df_test.drop(columns=[target_column])

    df_train_encoded = df_train.drop(
        columns=[target_column], errors="ignore"
    ).copy()
    df_test_encoded = df_test.copy()

    df_train_encoded = clean_column_names(df_train_encoded)
    df_test_encoded = clean_column_names(df_test_encoded)

    all_categorical_features = (
        binary_features
        + categorical_features
        + target_encoded_features
        + ordinal_features
    )

    placeholders = ["xna"]

    for feature in all_categorical_features:
        if feature in df_train_encoded.columns:
            df_train_encoded[feature].replace(
                placeholders, np.nan, inplace=True
            )
        if feature in df_test_encoded.columns:
            df_test_encoded[feature].replace(placeholders, np.nan, inplace=True)

    for feature in binary_features:
        clean_feature = clean_column_names(
            pd.DataFrame({feature: df_train_encoded[feature]})
        ).columns[0]

        if (
            clean_feature in df_train_encoded.columns
            and clean_feature in df_test_encoded.columns
        ):
            le = LabelEncoder()
            df_train_encoded[clean_feature] = le.fit_transform(
                df_train_encoded[clean_feature].astype(str)
            )

            df_test_encoded[clean_feature] = (
                df_test_encoded[clean_feature]
                .astype(str)
                .map({k: v for v, k in enumerate(le.classes_)})
                .fillna(-1)
                .astype(int)
            )
        else:
            print(
                f"Warning: Binary feature '{feature}' not found in both "
                "train and test data."
            )

    categorical_features_clean = [
        clean_column_names(
            pd.DataFrame({feat: df_train_encoded[feat]})
        ).columns[0]
        for feat in categorical_features
        if feat in df_train_encoded.columns
    ]

    if not categorical_features_clean:
        print(
            "Warning: No valid categorical features found for one-hot "
            "encoding."
        )

    if categorical_features_clean:
        ohe = OneHotEncoder(
            cols=categorical_features_clean,
            use_cat_names=True,
            handle_unknown="ignore",
        )
        df_train_encoded = ohe.fit_transform(df_train_encoded)
        df_test_encoded = ohe.transform(df_test_encoded)

        df_train_encoded = clean_column_names(df_train_encoded)
        df_test_encoded = clean_column_names(df_test_encoded)

        df_test_encoded = df_test_encoded.reindex(
            columns=df_train_encoded.columns, fill_value=0
        )

    if target_encoded_features:
        if not target_column:
            raise ValueError(
                "target_column must be specified when using "
                "target_encoded_features."
            )
        if target_column not in df_train.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in "
                "training data."
            )

        target_encoded_clean = [
            clean_column_names(
                pd.DataFrame({feat: df_train_encoded[feat]})
            ).columns[0]
            for feat in target_encoded_features
            if feat in df_train_encoded.columns
        ]

        loo_encoder = LeaveOneOutEncoder(
            cols=target_encoded_clean, handle_unknown="value", sigma=0.0
        )
        df_train_encoded[target_encoded_clean] = loo_encoder.fit_transform(
            df_train_encoded[target_encoded_clean], df_train[target_column]
        )
        df_test_encoded[target_encoded_clean] = loo_encoder.transform(
            df_test_encoded[target_encoded_clean]
        )

    if ordinal_features:
        if not ordinal_orders:
            raise ValueError(
                "ordinal_orders must be provided when using "
                "ordinal_features."
            )
        missing_orders = [
            feat for feat in ordinal_features if feat not in ordinal_orders
        ]
        if missing_orders:
            raise ValueError(
                f"Missing ordinal_orders for features: " f"{missing_orders}"
            )

        ordinal_features_clean = [
            clean_column_names(
                pd.DataFrame({feat: df_train_encoded[feat]})
            ).columns[0]
            for feat in ordinal_features
            if feat in df_train_encoded.columns
        ]

        ordinal_categories = [
            ordinal_orders[feat]
            for feat in ordinal_features
            if feat in ordinal_orders
        ]
        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df_train_encoded[ordinal_features_clean] = (
            ordinal_encoder.fit_transform(
                df_train_encoded[ordinal_features_clean]
            )
        )
        df_test_encoded[ordinal_features_clean] = ordinal_encoder.transform(
            df_test_encoded[ordinal_features_clean]
        )

    if target_column and target_column in df_train.columns:
        df_train_encoded["target"] = df_train[target_column].values

    unwanted_suffixes = ["_xna", "-1"]
    final_columns = [
        col
        for col in df_train_encoded.columns
        if not any(suffix in col for suffix in unwanted_suffixes)
    ]

    if target_column.lower() in final_columns:
        final_columns.remove(target_column.lower())

    if target_column and target_column in df_train.columns:
        df_train_encoded = df_train_encoded[final_columns + ["target"]]
    else:
        df_train_encoded = df_train_encoded[final_columns]

    df_test_encoded = df_test_encoded[final_columns]

    constant_cols = [
        col
        for col in df_train_encoded.columns
        if df_train_encoded[col].nunique() == 1
    ]
    if constant_cols:
        df_train_encoded.drop(columns=constant_cols, inplace=True)
        df_test_encoded.drop(
            columns=constant_cols, inplace=True, errors="ignore"
        )

    return df_train_encoded, df_test_encoded
