"""
This module provides various functions for data preprocessing, anomaly
detection, feature engineering, and statistical analysis, specifically
tailored for machine learning workflows.

Functions included:
- `detect_anomalies_iqr`: Detects anomalies in multiple features using the
  Interquartile Range (IQR) method.
- `flag_anomalies`: Flags anomalies in specified features using the IQR method.
- `calculate_cramers_v`: Computes Cramer's V statistic for
  categorical-categorical association.
- `handle_missing_values`: Handles missing data by dropping columns or rows
  based on a threshold.
- `simple_imputation`: Performs simple imputation on missing values in
  training and testing datasets.
- `engineer_spaceship_features`: Performs feature engineering, particularly
  for a spaceship passenger dataset.
- `confidence_interval`: Calculates the confidence interval for a given
  dataset.
- `create_pipeline`: Creates a scikit-learn pipeline for preprocessing and
  modeling.

This module is intended for use in data preprocessing and feature engineering,
with functions designed to handle common tasks in machine learning pipelines,
such as anomaly detection, missing data handling, and feature engineering.
"""

from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


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
    correlation_threshold: float = 0.05,
    essential_features: List[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Reduces features based on missing values, variance, and correlation.

    This function performs feature reduction on the input DataFrames based on
    missing values, variance, and correlation with the target variable, while
    ensuring that essential features are always retained.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        target_col: The name of the target variable column.
        missing_threshold: Max allowable missing value ratio. Defaults to 0.5.
        variance_threshold: Min variance required. Defaults to 0.01.
        correlation_threshold: Min absolute correlation with target.
            Defaults to 0.05.
        essential_features: List of features to always keep, regardless of
            reduction criteria. Defaults to None.

    Returns:
        A tuple containing the reduced train and test DataFrames.

    Raises:
        ValueError: If the target column is not in the training DataFrame.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    essential_features = essential_features or []

    combined_df = pl.concat([train_df.drop(target_col), test_df])
    total_rows = len(combined_df)

    cols_to_keep_missing = _filter_by_missing_values(
        combined_df.drop(essential_features), total_rows, missing_threshold
    )
    filtered_df = combined_df.select(cols_to_keep_missing + essential_features)

    numeric_cols = _get_numeric_columns(filtered_df)
    numeric_cols = [col for col in numeric_cols if col not in essential_features]

    cols_to_keep_variance = _filter_by_variance(
        filtered_df, numeric_cols, variance_threshold
    )

    cols_to_keep_correlation = _filter_by_correlation(
        train_df, numeric_cols, target_col, correlation_threshold
    )

    final_cols = list(
        set(cols_to_keep_missing)
        & set(cols_to_keep_variance)
        & set(cols_to_keep_correlation)
    )
    final_cols.extend([col for col in cols_to_keep_missing if col not in numeric_cols])
    final_cols.extend(essential_features)

    return (
        train_df.select([target_col] + final_cols),
        test_df.select(final_cols),
    )


def impute_numerical_features(train_df, test_df, target_col):
    """Imputes missing values in numerical features using KNNImputer.

    Args:
        train_df: Polars DataFrame for the training set.
        test_df: Polars DataFrame for the test set.
        target_col: Name of the target column in the training set.

    Returns:
        Tuple of imputed training and test DataFrames.
    """
    if not train_df.shape[0] or not test_df.shape[0]:
        print("Warning: One of the DataFrames is empty. Skipping numerical imputation.")
        return train_df, test_df

    numerical_features = [
        col
        for col, dtype in zip(train_df.columns, train_df.dtypes)
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
        and col != target_col
    ]

    if not numerical_features:
        print("Warning: No numerical features found. Skipping numerical imputation.")
        return train_df, test_df

    imputer = KNNImputer(n_neighbors=5)

    train_target = train_df[target_col]
    train_features = train_df.drop(target_col)

    train_numerical_pd = train_features[numerical_features].to_pandas()
    test_numerical_pd = test_df[numerical_features].to_pandas()

    train_numerical_imputed = imputer.fit_transform(train_numerical_pd)
    test_numerical_imputed = imputer.transform(test_numerical_pd)

    train_numerical_imputed_df = pl.DataFrame(
        train_numerical_imputed, schema=numerical_features
    )
    test_numerical_imputed_df = pl.DataFrame(
        test_numerical_imputed, schema=numerical_features
    )

    train_df = (
        train_features.drop(numerical_features)
        .hstack(train_numerical_imputed_df)
        .with_columns(train_target)
    )

    test_df = test_df.drop(numerical_features).hstack(test_numerical_imputed_df)

    return train_df, test_df


def impute_categorical_features(train_df, test_df, target_col):
    """Imputes missing values in categorical features using the mode.

    Args:
        train_df: Polars DataFrame for the training set.
        test_df: Polars DataFrame for the test set.
        target_col: Name of the target column in the training set.

    Returns:
        Tuple of imputed training and test DataFrames.
    """
    if not train_df.shape[0] or not test_df.shape[0]:
        print(
            "Warning: One of the DataFrames is empty. Skipping categorical imputation."
        )
        return train_df, test_df

    categorical_features = [
        col
        for col, dtype in zip(train_df.columns, train_df.dtypes)
        if dtype == pl.Categorical and col != target_col
    ]

    if not categorical_features:
        print("Warning: No categorical features found. Skipping imputation.")
        return train_df, test_df

    for col in categorical_features:
        train_df = train_df.with_columns(pl.col(col).fill_null("mode"))
        test_df = test_df.with_columns(pl.col(col).fill_null("mode"))

    return train_df, test_df


def count_duplicated_rows(dataframe: pl.DataFrame) -> None:
    """
    Count and print the number of duplicated rows in a Polars DataFrame
    (based on all columns).
    """
    num_duplicated_rows = dataframe.is_duplicated().sum()
    print(f"The DataFrame contains {num_duplicated_rows} duplicated rows.")


def _filter_by_missing_values(
    df: pl.DataFrame, total_rows: int, threshold: float
) -> List[str]:
    """Filters columns based on missing value ratio.

    Args:
        df: DataFrame to filter.
        total_rows: Total number of rows in the DataFrame.
        threshold: Maximum allowed ratio of missing values.

    Returns:
        List of column names that pass the missing value filter.
    """
    missing_ratios = df.null_count() / total_rows
    return [
        col
        for col, ratio in zip(df.columns, missing_ratios.to_numpy()[0])
        if ratio <= threshold
    ]


def _get_numeric_columns(df: pl.DataFrame) -> List[str]:
    """Returns a list of numeric column names.

    Args:
        df: DataFrame to analyze.

    Returns:
        List of column names with numeric data types.
    """
    numeric_types = (
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    )
    return [col for col in df.columns if df[col].dtype in numeric_types]


def _filter_by_variance(
    df: pl.DataFrame, columns: List[str], threshold: float
) -> List[str]:
    """Filters columns based on variance.

    Args:
        df: DataFrame containing the columns to filter.
        columns: List of column names to consider.
        threshold: Minimum required variance.

    Returns:
        List of column names that pass the variance filter.
    """
    variances = df.select(columns).var().to_dict(as_series=False)
    return [col for col, var in variances.items() if var[0] > threshold]


def _filter_by_correlation(
    df: pl.DataFrame, columns: List[str], target_col: str, threshold: float
) -> List[str]:
    """Filters columns based on correlation with the target variable.

    Args:
        df: DataFrame containing the columns and target variable.
        columns: List of column names to consider.
        target_col: Name of the target column.
        threshold: Minimum required absolute correlation.

    Returns:
        List of column names that pass the correlation filter.
    """

    def correlation_with_target(col: str) -> float:
        """Calculates absolute Pearson correlation with the target."""
        x = df[col].to_numpy()
        y = df[target_col].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        return np.abs(np.corrcoef(x[mask], y[mask])[0, 1])

    correlations = {col: correlation_with_target(col) for col in columns}
    return ["amt_income_total"] + [
        col
        for col, corr in correlations.items()
        if corr > threshold and col != "amt_income_total"
    ]


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
    Identify and flag anomalies in a DataFrame based on the Interquartile Range
    (IQR) method for specified features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        features (List[str]): A list of column names in the DataFrame to check for
        anomalies.

    Returns:
        pd.Series: A Series of boolean values where True indicates an anomaly in
        any of the specified features.
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
    """Calculates the percentage of missing values for each column in a Polars DataFrame
    and returns the top N columns with the highest percentage of missing values.

    Args:
        df: The Polars DataFrame to analyze for missing values.
        top_n: The number of top columns to return (default is 5).

    Returns:
        pl.DataFrame: A Polars DataFrame with columns for column names and missing
        value percentages.
    """
    total_rows = df.height
    missing_counts = df.null_count().row(0)
    missing_percentages = [
        {"column": col, "missing_percentage": count / total_rows * 100}
        for col, count in zip(df.columns, missing_counts)
        if count > 0
    ]

    if missing_percentages:
        missing_df = pl.DataFrame(missing_percentages)
        return (
            missing_df.sort("missing_percentage", descending=True)
            .with_columns(pl.col("missing_percentage").round(2))
            .head(top_n)
        )
    else:
        return pl.DataFrame()


def analyze_missing_values(
    train_df: pl.DataFrame, test_df: pl.DataFrame, top_n: int = 5
) -> None:
    """Analyzes and prints the top N columns with the highest percentage of missing values
    for both train and test DataFrames.

    Args:
        train_df: The training Polars DataFrame.
        test_df: The testing Polars DataFrame.
        top_n: The number of top columns to display (default is 5).
    """
    train_missing = get_top_missing_value_percentages(train_df, top_n)
    test_missing = get_top_missing_value_percentages(test_df, top_n)

    if train_missing.is_empty():
        print("No missing values found in the reduced train set.")
    else:
        print(f"Top {top_n} columns with missing values in reduced train set:")
        print(train_missing)

    if test_missing.is_empty():
        print("No missing values found in the reduced test set.")
    else:
        print(f"\nTop {top_n} columns with missing values in reduced test set:")
        print(test_missing)


def confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculates the confidence interval for a given dataset.

    Args:
        data (List[float]): A list of numerical data points.
        confidence (float): The confidence level for the interval. Defaults to 0.95.

    Returns:
        Tuple[float, float, float]: A tuple containing the mean, lower bound, and
        upper bound of the confidence interval.
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
        preprocessor (sklearn.base.TransformerMixin): The preprocessing component
        of the pipeline.
        model (sklearn.base.BaseEstimator): The classifier component of the
        pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn Pipeline object that sequentially
        applies the preprocessor and the classifier.
    """

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def create_stratified_sample(
    data: pl.DataFrame,
    target_column: str,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Creates a stratified sample from a Polars DataFrame, preserving class
    proportions of the target variable.

    Args:
        data: The Polars DataFrame to sample from.
        target_column: The name of the target variable column.
        sample_size: The desired sample size.
        random_state: Seed for the random number generator (for reproducibility).

    Returns:
        A Pandas DataFrame containing the stratified sample.
    """
    features_df = data.drop(target_column)
    target_series = data[target_column]

    _, sample_features, _, sample_target = train_test_split(
        features_df.to_pandas(),
        target_series.to_pandas(),
        test_size=sample_size,
        stratify=target_series.to_pandas(),
        random_state=random_state,
    )

    return pd.concat([sample_features, sample_target], axis=1)
