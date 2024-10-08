"""
This module provides various functions for data preprocessing, anomaly
detection, feature engineering, and statistical analysis, specifically
tailored for machine learning workflows.

Functions included:
- `reduce_memory_usage_pl`: Optimizes memory usage of Polars DataFrames.
- `detect_anomalies_iqr`: Detects anomalies in features using the IQR method.
- `flag_anomalies`: Flags anomalies in specified features using the IQR method.
- `calculate_cramers_v`: Computes Cramer's V statistic for categorical association.
- `handle_missing_values`: Handles missing data by dropping columns or rows.
- `impute_numerical_features`: Performs KNN imputation on numerical values.
- `impute_categorical_features`: Performs mode imputation on categorical values.
- `count_duplicated_rows`: Counts duplicated rows in a Polars DataFrame.
- `get_top_missing_value_percentages`: Retrieves top N columns with highest
  missing percentages.
- `analyze_missing_values`: Analyzes and prints top N missing value columns.
- `confidence_interval`: Calculates the confidence interval for a dataset.
- `create_pipeline`: Creates a scikit-learn pipeline for preprocessing and modeling.
- `create_stratified_sample`: Creates a stratified sample from a Polars DataFrame.
- `calculate_cliff_delta`: Calculates Cliff's delta, a non-parametric effect size.
- `initial_feature_reduction`: Reduces features based on missing values,
  variance, and correlation.

This module is intended for use in data preprocessing and feature engineering,
with functions designed to handle common tasks in machine learning pipelines,
such as anomaly detection, missing data handling, and feature engineering.
"""

from collections import Counter
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


def reduce_memory_usage_pl(
    df: pl.DataFrame, verbose: bool = True
) -> pl.DataFrame:
    """Reduces memory usage of a Polars DataFrame by optimizing data types.

    Attempts to downcast numeric columns to the smallest possible data type
    that can represent the data without loss of information. Also converts
    string columns to categorical type.

    Args:
        df: A Polars DataFrame to optimize.
        verbose: If True, prints memory usage before and after optimization.

    Returns:
        A Polars DataFrame with optimized memory usage.
    """
    if verbose:
        print(f"Size before reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Initial data types: {Counter(df.dtypes)}")

    numeric_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    numeric_float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        col_type = df[col].dtype

        if col_type in numeric_int_types:
            c_min = df[col].min().item()
            c_max = df[col].max().item()

            if np.iinfo(np.int8).min <= c_min <= c_max <= np.iinfo(np.int8).max:
                new_type = pl.Int8
            elif np.iinfo(np.int16).min <= c_min <= c_max <= np.iinfo(
                np.int16
            ).max:
                new_type = pl.Int16
            elif np.iinfo(np.int32).min <= c_min <= c_max <= np.iinfo(
                np.int32
            ).max:
                new_type = pl.Int32
            else:
                new_type = pl.Int64

            if new_type != col_type:
                df = df.with_columns(pl.col(col).cast(new_type))

        elif col_type in numeric_float_types:
            c_min = df[col].min().item()
            c_max = df[col].max().item()
            if (
                np.finfo(np.float32).min <= c_min <= c_max <= np.finfo(
                    np.float32
                ).max
            ):
                df = df.with_columns(pl.col(col).cast(pl.Float32))

        elif col_type == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

    if verbose:
        print(f"Size after reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Final data types: {Counter(df.dtypes)}")

    return df


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
    missing_counts = df.null_count()
    missing_ratios = {
        col: count / total_rows for col, count in zip(df.columns, missing_counts.row(0))
    }
    return [
        col for col, ratio in missing_ratios.items() if ratio <= threshold
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
    variances = df.select(columns).var()
    variances_dict = {
        col: var[0] for col, var in variances.to_dict(as_series=False).items()
    }
    return [col for col, var in variances_dict.items() if var > threshold]


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
        x = df[col].to_numpy().astype(float)
        y = df[target_col].to_numpy().astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0

    correlations = {col: correlation_with_target(col) for col in columns}
    return [col for col, corr in correlations.items() if corr >= threshold]


def initial_feature_reduction(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
    missing_threshold: float = 0.3,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.05,
    essential_features: List[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Reduces features based on missing values, variance, and correlation.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        target_col: The name of the target variable column.
        missing_threshold: Max allowable missing value ratio. Defaults to 0.3.
        variance_threshold: Min variance required. Defaults to 0.01.
        correlation_threshold: Min absolute correlation with target.
            Defaults to 0.05.
        essential_features: List of features to always keep. Defaults to None.

    Returns:
        A tuple containing the reduced train and test DataFrames.

    Raises:
        ValueError: If the target column is not in the training DataFrame.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    essential_features = essential_features or []

    combined_df = pl.concat([train_df.drop(target_col), test_df])
    total_rows = combined_df.height

    cols_missing = _filter_by_missing_values(
        combined_df.drop(essential_features), total_rows, missing_threshold
    )
    filtered_df = combined_df.select(cols_missing + essential_features)

    numeric_cols = _get_numeric_columns(filtered_df)
    numeric_cols = [col for col in numeric_cols if col not in essential_features]

    cols_variance = _filter_by_variance(filtered_df, numeric_cols, variance_threshold)
    cols_correlation = _filter_by_correlation(
        train_df, numeric_cols, target_col, correlation_threshold
    )

    final_numeric_cols = list(
        set(cols_missing) & set(cols_variance) & set(cols_correlation)
    )

    non_numeric_cols = [
        col for col in cols_missing if col not in numeric_cols
    ]

    final_cols = final_numeric_cols + non_numeric_cols + essential_features

    seen = set()
    final_cols = [x for x in final_cols if not (x in seen or seen.add(x))]

    reduced_train = train_df.select([target_col] + final_cols)
    reduced_test = test_df.select(final_cols)

    return reduced_train, reduced_test


def impute_numerical_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, target_col: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Imputes missing values in numerical features using KNNImputer.

    Args:
        train_df: Polars DataFrame for the training set.
        test_df: Polars DataFrame for the test set.
        target_col: Name of the target column in the training set.

    Returns:
        Tuple of imputed training and test DataFrames.
    """
    if train_df.is_empty() or test_df.is_empty():
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

    train_numerical_pd = train_features.select(numerical_features).to_pandas()
    test_numerical_pd = test_df.select(numerical_features).to_pandas()

    imputer.fit(train_numerical_pd)
    train_numerical_imputed = imputer.transform(train_numerical_pd)
    test_numerical_imputed = imputer.transform(test_numerical_pd)

    train_numerical_imputed_df = pl.from_pandas(
        pd.DataFrame(train_numerical_imputed, columns=numerical_features)
    )
    test_numerical_imputed_df = pl.from_pandas(
        pd.DataFrame(test_numerical_imputed, columns=numerical_features)
    )

    train_df = (
        train_features.drop(numerical_features)
        .hstack(train_numerical_imputed_df)
        .with_columns(pl.Series(name=target_col, values=train_target))
    )

    test_df = test_df.drop(numerical_features).hstack(test_numerical_imputed_df)

    return train_df, test_df


def impute_categorical_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, target_col: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Imputes missing values in categorical features using the mode.

    Args:
        train_df: Polars DataFrame for the training set.
        test_df: Polars DataFrame for the test set.
        target_col: Name of the target column in the training set.

    Returns:
        Tuple of imputed training and test DataFrames.
    """
    if train_df.is_empty() or test_df.is_empty():
        print(
            "Warning: One of the DataFrames is empty. "
            "Skipping categorical imputation."
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
        value_counts_df = train_df[col].value_counts()
        mode_value = (
            value_counts_df
            .sort("count", descending=True)
            .get_column(col)[0]
        )
        if pd.isnull(mode_value):
            mode_value = "Unknown"

        train_df = train_df.with_columns(pl.col(col).fill_null(mode_value))
        test_df = test_df.with_columns(pl.col(col).fill_null(mode_value))

    return train_df, test_df


def count_duplicated_rows(dataframe: pl.DataFrame) -> None:
    """Counts and prints the number of duplicated rows in a Polars DataFrame."""
    num_duplicated_rows = dataframe.filter(dataframe.is_duplicated()).height
    print(f"The DataFrame contains {num_duplicated_rows} duplicated rows.")


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect anomalies in.

    Returns:
        DataFrame containing the anomalies for each feature.
    """
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical. Skipping.")
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
            anomalies_list.append(feature_anomalies)
        else:
            print(f"No anomalies detected in feature '{feature}'.")

    if anomalies_list:
        anomalies = pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        anomalies = anomalies[features]
    else:
        anomalies = pd.DataFrame(columns=features)

    return anomalies


def flag_anomalies(df: pd.DataFrame, features: List[str]) -> pd.Series:
    """Flags anomalies in specified features using the IQR method.

    Args:
        df: The input DataFrame.
        features: List of feature names to check for anomalies.

    Returns:
        A Series of boolean values where True indicates an anomaly.
    """
    anomaly_flags = pd.Series(False, index=df.index)

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical. Skipping.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        feature_anomalies = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        anomaly_flags |= feature_anomalies

    return anomaly_flags


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculates Cramer's V statistic for categorical-categorical association.

    Args:
        x: Categorical feature.
        y: Categorical target or feature.

    Returns:
        Cramer's V statistic.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if rcorr == 0 or kcorr == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def get_top_missing_value_percentages(
    df: pl.DataFrame, top_n: int = 5
) -> pl.DataFrame:
    """Retrieves top N columns with highest missing value percentages.

    Args:
        df: The Polars DataFrame to analyze.
        top_n: The number of top columns to return.

    Returns:
        DataFrame with columns for names and missing value percentages.
    """
    total_rows = df.height
    missing_counts = df.null_count().row(0)
    missing_percentages = [
        {"column": col, "missing_percentage": (count / total_rows) * 100}
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
    """Analyzes and prints top N missing value columns for train and test sets.

    Args:
        train_df: The training Polars DataFrame.
        test_df: The testing Polars DataFrame.
        top_n: The number of top columns to display.
    """
    train_missing = get_top_missing_value_percentages(train_df, top_n)
    test_missing = get_top_missing_value_percentages(test_df, top_n)

    if not train_missing.is_empty():
        print(f"Top {top_n} columns with missing values in train set:")
        print(train_missing)
    else:
        print("No missing values found in the train set.")

    if not test_missing.is_empty():
        print(f"\nTop {top_n} columns with missing values in test set:")
        print(test_missing)
    else:
        print("No missing values found in the test set.")


def confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculates the confidence interval for a given dataset.

    Args:
        data: A list of numerical data points.
        confidence: The confidence level for the interval.

    Returns:
        A tuple containing the mean, lower bound, and upper bound.
    """
    array_data = np.array(data, dtype=float)
    sample_size = len(array_data)
    mean_value = np.mean(array_data)
    standard_error = stats.sem(array_data)
    if sample_size < 2:
        return mean_value, np.nan, np.nan
    margin_of_error = standard_error * stats.t.ppf(
        (1 + confidence) / 2.0, sample_size - 1
    )
    return (
        mean_value,
        mean_value - margin_of_error,
        mean_value + margin_of_error,
    )


def create_pipeline(preprocessor: Pipeline, model: Pipeline) -> Pipeline:
    """Creates a machine learning pipeline with a preprocessor and a classifier.

    Args:
        preprocessor: The preprocessing component of the pipeline.
        model: The classifier component of the pipeline.

    Returns:
        A scikit-learn Pipeline object.
    """
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def create_stratified_sample(
    data: pl.DataFrame,
    target_column: str,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Creates a stratified sample from a Polars DataFrame.

    Args:
        data: The Polars DataFrame to sample from.
        target_column: The name of the target variable column.
        sample_size: The desired sample size.
        random_state: Seed for reproducibility.

    Returns:
        A Pandas DataFrame containing the stratified sample.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    total_rows = len(data)
    if sample_size > total_rows:
        raise ValueError("Sample size cannot exceed total number of rows.")

    fraction = sample_size / total_rows
    sampled_df = data.to_pandas().groupby(target_column, group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=random_state)
    )
    return sampled_df.reset_index(drop=True)


def calculate_cliff_delta(
    df: pd.DataFrame, feature: str, target: str
) -> Union[float, np.float64]:
    """Calculates Cliff's delta, a non-parametric effect size measure.

    Args:
        df: The input DataFrame containing the feature and target columns.
        feature: The name of the numerical feature column.
        target: The name of the binary target column (0 or 1).

    Returns:
        Cliff's delta, a value between -1 and 1.
    """
    group1 = df[df[target] == 0][feature].dropna().values
    group2 = df[df[target] == 1][feature].dropna().values

    n1 = len(group1)
    n2 = len(group2)

    if n1 == 0 or n2 == 0:
        return np.nan

    greater = np.sum(group1[:, None] > group2)
    less = np.sum(group1[:, None] < group2)

    cliff_d = (greater - less) / (n1 * n2)
    return cliff_d
