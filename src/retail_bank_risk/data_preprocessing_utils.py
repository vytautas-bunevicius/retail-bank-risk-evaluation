# cspell:disable
# pylint:disable=line-too-long
"""
This module provides various functions for data preprocessing, anomaly detection, feature engineering,
and statistical analysis, specifically tailored for machine learning workflows.

Functions included:
- `detect_anomalies_iqr`: Detects anomalies in multiple features using the Interquartile Range (IQR) method.
- `flag_anomalies`: Flags anomalies in specified features using the IQR method.
- `calculate_cramers_v`: Computes Cramer's V statistic for categorical-categorical association.
- `handle_missing_values`: Handles missing data by dropping columns or rows based on a threshold.
- `simple_imputation`: Performs simple imputation on missing values in training and testing datasets.
- `engineer_spaceship_features`: Performs feature engineering, particularly for a spaceship passenger dataset.
- `confidence_interval`: Calculates the confidence interval for a given dataset.
- `create_pipeline`: Creates a scikit-learn pipeline for preprocessing and modeling.

This module is intended for use in data preprocessing and feature engineering, with functions designed to handle common
tasks in machine learning pipelines, such as anomaly detection, missing data handling, and feature engineering.
"""


from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.pipeline import Pipeline


def reduce_memory_usage_pl(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Reduces memory usage of a Polars DataFrame by optimizing data types.

    This function attempts to downcast numeric columns to the smallest possible
    data type that can represent the data without loss of information. It also
    converts string columns to categorical type.

    Args:
        df: A Polars DataFrame to optimize.
        verbose: If True, print memory usage before and after optimization.

    Returns:
        A Polars DataFrame with optimized memory usage.

    References:
        Adapted from:
        https://www.kaggle.com/code/demche/polars-memory-usage-optimization
        Original pandas version:
        https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65
    """
    if verbose:
        print(f"Size before memory reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Initial data types: {Counter(df.dtypes)}")

    numeric_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    numeric_float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        col_type = df[col].dtype

        if col_type in numeric_int_types:
            c_min = df[col].min() * 10  # Prevent possible integer overflow
            c_max = df[col].max() * 10

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                new_type = pl.Int8
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                new_type = pl.Int16
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                new_type = pl.Int32
            else:
                new_type = pl.Int64

            df = df.with_columns(df[col].cast(new_type))

        elif col_type in numeric_float_types:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))

        elif col_type == pl.String:
            df = df.with_columns(df[col].cast(pl.Categorical))

    if verbose:
        print(f"Size after memory reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Final data types: {Counter(df.dtypes)}")

    return df


def initial_feature_reduction(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
    missing_threshold: float = 0.5,
    variance_threshold: float = 0.01,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Reduces features in both training and testing datasets based on missing
    value ratios and variance thresholds.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame.
        target_col (str): The name of the target variable column.
        missing_threshold (float): Maximum allowable missing value ratio for
            retaining a column (default 0.5).
        variance_threshold (float): Minimum variance required to retain a
            column (default 0.01).

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the reduced
            training and testing DataFrames.
    """
    combined_df = pl.concat([train_df.drop(target_col), test_df])
    total_rows = len(combined_df)
    missing_ratios = combined_df.null_count() / total_rows
    cols_to_keep = [
        col
        for col, ratio in zip(combined_df.columns, missing_ratios.to_numpy()[0])
        if ratio <= missing_threshold
    ]

    filtered_df = combined_df.select(cols_to_keep)
    numeric_cols = [
        col
        for col in filtered_df.columns
        if filtered_df[col].dtype in [pl.Float64, pl.Int64]
    ]
    variances = {col: filtered_df[col].var() for col in numeric_cols}
    variance_filtered_cols = [
        col for col, var in variances.items() if var > variance_threshold
    ]

    final_cols = variance_filtered_cols + [
        col for col in cols_to_keep if col not in numeric_cols
    ]
    reduced_train_df = train_df.select([target_col] + final_cols)
    reduced_test_df = test_df.select(final_cols)

    return reduced_train_df, reduced_test_df


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features (List[str]): List of features to detect anomalies in.

    Returns:
        pd.DataFrame: DataFrame containing the anomalies for each feature.
    """
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical and will be skipped.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        feature_anomalies = df[
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ]
        if not feature_anomalies.empty:
            print(f"Anomalies detected in feature '{feature}':")
            print(feature_anomalies)
        else:
            print(f"No anomalies detected in feature '{feature}'.")
        anomalies_list.append(feature_anomalies)

    if anomalies_list:
        anomalies = pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        anomalies = anomalies[features]
    else:
        anomalies = pd.DataFrame(columns=features)

    return anomalies


def flag_anomalies(df: pd.DataFrame, features: List[str]) -> pd.Series:
    """
    Identify and flag anomalies in a DataFrame based on the Interquartile Range (IQR) method for specified features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        features (List[str]): A list of column names in the DataFrame to check for anomalies.

    Returns:
        pd.Series: A Series of boolean values where True indicates an anomaly in any of the specified features.
    """
    anomaly_flags = pd.Series(False, index=df.index)

    for feature in features:
        first_quartile = df[feature].quantile(0.25)
        third_quartile = df[feature].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range

        feature_anomalies = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        anomaly_flags |= feature_anomalies

    return anomaly_flags


def calculate_cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.

    Args:
        x: pandas Series
        y: pandas Series

    Returns:
        float: Cramer's V
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


def get_top_missing_value_percentages(df: pl.DataFrame, top_n: int = 5) -> pl.DataFrame:
    """
    Calculates the percentage of missing values for each column in a Polars DataFrame
    and returns the top N columns with the highest percentage of missing values.

    Args:
        df: The Polars DataFrame to analyze for missing values.
        top_n: The number of top columns to return (default is 5).

    Returns:
        A Polars DataFrame with columns for column names and missing value percentages.
    """
    total_rows = df.height
    missing_counts = df.null_count().row(0)
    missing_percentages = [
        {"column": col, "missing_percentage": count / total_rows * 100}
        for col, count in zip(df.columns, missing_counts)
        if count > 0
    ]

    return (
        pl.DataFrame(missing_percentages, orient="row")
        .sort("missing_percentage", descending=True)
        .with_columns(pl.col("missing_percentage").round(2))
        .head(top_n)
    )


def analyze_missing_values(
    train_df: pl.DataFrame, test_df: pl.DataFrame, top_n: int = 5
) -> None:
    """
    Analyzes and prints the top N columns with the highest percentage of missing values
    for both train and test DataFrames.

    Args:
        train_df: The training Polars DataFrame.
        test_df: The testing Polars DataFrame.
        top_n: The number of top columns to display (default is 5).
    """
    train_missing = get_top_missing_value_percentages(train_df, top_n)
    test_missing = get_top_missing_value_percentages(test_df, top_n)

    print(f"Top {top_n} columns with missing values in reduced train set:")
    print(train_missing)

    print(f"\nTop {top_n} columns with missing values in reduced test set:")
    print(test_missing)


def confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculates the confidence interval for a given dataset.

    Args:
        data (List[float]): A list of numerical data points.
        confidence (float): The confidence level for the interval. Defaults to 0.95.

    Returns:
        Tuple[float, float, float]: A tuple containing the mean, lower bound, and upper bound of the confidence interval.
    """
    array_data = np.array(data, dtype=float)
    sample_size = len(array_data)
    mean_value = np.mean(array_data)
    standard_error = stats.sem(array_data)
    margin_of_error = standard_error * stats.t.ppf(
        (1 + confidence) / 2.0, sample_size - 1
    )
    return (
        mean_value,
        mean_value - margin_of_error,
        mean_value + margin_of_error,
    )


def create_pipeline(preprocessor: Pipeline, model: Pipeline) -> Pipeline:
    """
    Create a machine learning pipeline with a preprocessor and a classifier.

    Parameters:
    preprocessor (sklearn.base.TransformerMixin): The preprocessing component of the pipeline.
    model (sklearn.base.BaseEstimator): The classifier component of the pipeline.

    Returns:
    sklearn.pipeline.Pipeline: A scikit-learn Pipeline object that sequentially applies the preprocessor and the classifier.
    """
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])
