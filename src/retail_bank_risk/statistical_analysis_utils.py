"""
This module provides utility functions for statistical analysis of features
in a dataset, particularly for binary classification problems.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas.api.types as pdt
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm.notebook import tqdm


def vectorized_cliff_delta(group_one: np.ndarray, group_two: np.ndarray) -> float:
    """
    Calculate Cliff's Delta using vectorized operations.

    Cliff's Delta is a non-parametric effect size measure that quantifies the
    amount of difference between two groups of observations.

    Args:
        group_one: Array of values for the first group.
        group_two: Array of values for the second group.

    Returns:
        Cliff's Delta value (-1 to 1).
        0 indicates equality, 1 indicates group_one dominance, -1 indicates
        group_two dominance.
    """
    n1, n2 = len(group_one), len(group_two)
    diffs = group_one[:, None] - group_two
    return (np.sum(diffs > 0) - np.sum(diffs < 0)) / (n1 * n2)


def bootstrap_cliff_delta(
    data1: np.ndarray, data2: np.ndarray, num_iterations: int = 500
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for Cliff's Delta.

    Args:
        data1: Array of values for the first group.
        data2: Array of values for the second group.
        num_iterations: Number of bootstrap iterations.

    Returns:
        Lower and upper bounds of the 95% confidence interval.
    """
    n1, n2 = len(data1), len(data2)
    deltas = np.array(
        [
            vectorized_cliff_delta(
                np.random.choice(data1, size=n1, replace=True),
                np.random.choice(data2, size=n2, replace=True),
            )
            for _ in range(num_iterations)
        ]
    )
    return tuple(np.percentile(deltas, [2.5, 97.5]))


def bootstrap_cramers_v(
    contingency_table: pd.DataFrame, num_iterations: int = 500
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for Cramer's V.

    Args:
        contingency_table: Contingency table of observed frequencies.
        num_iterations: Number of bootstrap iterations.

    Returns:
        Lower and upper bounds of the 95% confidence interval.
    """
    n = contingency_table.sum().sum()
    min_dim = max(1, min(contingency_table.shape) - 1)

    cramers_v_values = []
    for _ in range(num_iterations):
        resampled_table = np.random.multinomial(
            n, contingency_table.values.flatten() / n
        ).reshape(contingency_table.shape)
        chi2 = stats.chi2_contingency(resampled_table)[0]
        cramers_v_values.append(np.sqrt(chi2 / (n * min_dim)))

    return tuple(np.percentile(cramers_v_values, [2.5, 97.5]))


def simulate_p_value(
    contingency_table: pd.DataFrame, num_simulations: int = 10000
) -> float:
    """
    Simulate p-value for contingency tables with low expected frequencies.

    Args:
        contingency_table: The observed contingency table.
        num_simulations: Number of simulations to perform.

    Returns:
        Simulated p-value.
    """
    observed = contingency_table.values
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total_sum = observed.sum()
    expected = np.outer(row_sums, col_sums) / total_sum
    chi2_observed = np.sum(
        np.divide(
            (observed - expected) ** 2,
            expected,
            out=np.zeros_like(expected),
            where=expected != 0,
        )
    )

    chi2_simulated = np.zeros(num_simulations)
    for i in range(num_simulations):
        simulated = np.random.multinomial(
            total_sum, (observed / total_sum).flatten()
        ).reshape(observed.shape)
        expected_sim = (
            np.outer(simulated.sum(axis=1), simulated.sum(axis=0))
            / simulated.sum()
        )
        chi2_simulated[i] = np.sum(
            np.divide(
                (simulated - expected_sim) ** 2,
                expected_sim,
                out=np.zeros_like(expected_sim),
                where=expected_sim != 0,
            )
        )

    p_value = (chi2_simulated >= chi2_observed).mean()
    return p_value


def _analyze_numerical_feature(
    feature: str, data: pd.DataFrame, target: str
) -> Tuple[str, Optional[Dict]]:
    """
    Analyze a numerical feature using Mann-Whitney U and Cliff's Delta.

    Args:
        feature: Name of the feature.
        data: DataFrame containing feature and target.
        target: Name of the target variable.

    Returns:
        Tuple containing feature name and a dictionary of results, or None if
        analysis fails.
    """
    feature_data = data[feature]
    target_data = data[target]

    if not pdt.is_numeric_dtype(feature_data):
        print(f"Error: Feature '{feature}' is not numeric.")
        return feature, None

    total_count = len(feature_data)
    na_count = feature_data.isna().sum()
    unique_values = feature_data.nunique()

    unique_targets = target_data.unique()
    if len(unique_targets) != 2:
        print(
            f"Error: Target variable '{target}' is not binary. "
            f"Unique values: {unique_targets}"
        )
        return feature, None

    try:
        target_data = pd.to_numeric(target_data, errors="coerce")
        if target_data.isna().any():
            raise ValueError(
                f"Target variable '{target}' contains non-numeric values."
            )
    except ValueError as e:
        print(f"{e}")
        return feature, None

    unique_targets_numeric = target_data.dropna().unique()
    if len(unique_targets_numeric) != 2:
        print(
            f"Error: Target variable '{target}' is not binary after "
            f"converting to numeric. Unique numeric values: "
            f"{unique_targets_numeric}"
        )
        return feature, None

    if not all(val in [0, 1] for val in unique_targets_numeric):
        try:
            target_data = target_data.map(
                {unique_targets_numeric[0]: 0, unique_targets_numeric[1]: 1}
            )
        except (TypeError, ValueError) as e:
            print(f"Error mapping target values: {e}")
            return feature, None

    group_zero = feature_data[target_data == 0]
    group_one = feature_data[target_data == 1]

    group_zero_count = len(group_zero)
    group_one_count = len(group_one)

    group_zero_na_count = group_zero.isna().sum()
    group_one_na_count = group_one.isna().sum()

    if group_zero_count == 0 and group_one_count == 0:
        error_msg = (
            f"Analysis failed for feature {feature}: Both groups have zero "
            f"samples. This might indicate an issue with the target variable."
        )
        print(error_msg)
        return feature, None
    elif group_zero_count == 0 or group_one_count == 0:
        error_msg = (
            f"Analysis failed for feature {feature}:\n"
            f"Total samples: {total_count}, NA count: {na_count}, "
            f"Unique values: {unique_values}\n"
            f"Group 0: {group_zero_count} samples, {group_zero_na_count} NAs\n"
            f"Group 1: {group_one_count} samples, {group_one_na_count} NAs\n"
            f"One group has zero samples. This might indicate an imbalanced "
            f"dataset or an issue with the target variable."
        )
        print(error_msg)
        return feature, None

    try:
        group_zero_valid = group_zero.dropna().values
        group_one_valid = group_one.dropna().values

        if len(group_zero_valid) == 0 or len(group_one_valid) == 0:
            error_msg = (
                f"Analysis failed for feature {feature}: One or both groups "
                f"have zero size after dropping NAs.\n"
                f"Group 0: {len(group_zero_valid)} valid samples\n"
                f"Group 1: {len(group_one_valid)} valid samples"
            )
            print(error_msg)
            return feature, None

        _, p_value = stats.mannwhitneyu(
            group_zero_valid, group_one_valid, alternative="two-sided"
        )
        effect_size = vectorized_cliff_delta(group_zero_valid, group_one_valid)
        ci = bootstrap_cliff_delta(group_zero_valid, group_one_valid)

        return feature, {
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": ci,
            "test": "mannwhitneyu",
        }
    except (ValueError, TypeError) as e:
        print(f"Analysis failed for feature {feature}: {str(e)}")
        return feature, None


def _analyze_categorical_feature(
    feature: str, data: pd.DataFrame, target: str, auto_correct: bool = False
) -> Tuple[str, Optional[Dict]]:
    """
    Analyze a categorical feature using Chi-squared, Fisher's Exact, or
    simulated p-values.

    Args:
        feature: Name of the feature.
        data: DataFrame containing feature and target.
        target: Name of the target variable.
        auto_correct: If True and low cell counts for chi2 detected, switch
                      to simulated p-value.

    Returns:
        Tuple containing feature name and a dictionary of results, or None if
        analysis fails.
    """
    feature_data = data[feature].copy()
    target_data = data[target]

    unique_targets = target_data.unique()
    if len(unique_targets) != 2:
        print(
            f"Error: Target variable '{target}' is not binary. "
            f"Unique values: {unique_targets}"
        )
        return feature, None

    try:
        target_data = pd.to_numeric(target_data, errors="coerce")
        if target_data.isna().any():
            raise ValueError(
                f"Target variable '{target}' contains non-numeric values "
                f"that cannot be converted."
            )
    except ValueError as e:
        print(f"{e}")
        return feature, None

    unique_targets_numeric = target_data.dropna().unique()
    if len(unique_targets_numeric) != 2:
        print(
            f"Error: Target variable '{target}' is not binary after coercing "
            f"to numeric. Unique coerced values: {unique_targets_numeric}"
        )
        return feature, None

    if not all(val in [0, 1] for val in unique_targets_numeric):
        try:
            target_data = target_data.map(
                {unique_targets_numeric[0]: 0, unique_targets_numeric[1]: 1}
            )
        except (TypeError, ValueError) as e:
            print(f"Error mapping target values: {e}")
            return feature, None

    value_counts = feature_data.value_counts()
    rare_threshold = max(len(feature_data) * 0.01, 5)

    if len(value_counts) > 10:
        top_categories = value_counts.nlargest(9).index.tolist()
        feature_data = feature_data.map(
            lambda x: x if x in top_categories else "Other"
        )
    else:
        rare_categories = value_counts[value_counts < rare_threshold].index
        feature_data = feature_data.map(
            lambda x: x if x not in rare_categories else "Other"
        )

    try:
        contingency_table = pd.crosstab(feature_data, target_data)

        if np.any(contingency_table.values == 0):
            p_value = simulate_p_value(contingency_table)
            test_used = "chi2_simulated"
            chi2 = np.nan
            warnings.warn(
                f"Zero observed values in contingency table. Using simulated "
                f"p-value for {feature}."
            )
        else:
            chi2, p_value, _, expected = stats.chi2_contingency(
                contingency_table
            )

            if contingency_table.shape == (2, 2) and np.any(expected < 5):
                _, p_value = stats.fisher_exact(contingency_table)
                test_used = "fisher"
            elif np.any(expected < 1) or (
                auto_correct
                and np.any(expected < 5)
                and contingency_table.size > 4
            ):
                p_value = simulate_p_value(contingency_table)
                test_used = "chi2_simulated"
            elif np.any(expected < 5) and contingency_table.size > 4:
                warnings.warn(
                    f"Low expected cell counts in contingency table for "
                    f"{feature} may affect Chi-square reliability. Consider "
                    f"combining categories or setting `auto_correct=True`."
                )
                test_used = "chi2"
            else:
                test_used = "chi2"

        n = contingency_table.sum().sum()
        min_dim = max(1, min(contingency_table.shape) - 1)
        cramers_v = (
            np.sqrt(chi2 / (n * min_dim)) if not np.isnan(chi2) else np.nan
        )
        ci = bootstrap_cramers_v(contingency_table)

        return feature, {
            "p_value": p_value,
            "effect_size": cramers_v,
            "confidence_interval": ci,
            "test": test_used,
        }

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        print(f"Analysis failed for feature {feature}: {str(e)}")
        return feature, None


def analyze_feature(
    feature: str, data: pd.DataFrame, target: str, auto_correct=False
) -> Tuple[str, Optional[Dict]]:
    """
    Analyze a single feature. Handles boolean, numerical, and categorical data.
    """
    if pdt.is_bool_dtype(data[feature]):
        data[feature] = data[feature].astype(int)
        return _analyze_numerical_feature(feature, data, target)
    elif pdt.is_numeric_dtype(data[feature]):
        return _analyze_numerical_feature(feature, data, target)
    else:
        return _analyze_categorical_feature(
            feature, data, target, auto_correct=auto_correct
        )


def diagnose_target(data: pd.DataFrame, target: str) -> None:
    """
    Diagnose potential issues with the target variable.
    """
    target_data = data[target]
    print(f"Target variable diagnostics for '{target}':")
    print(f"Total samples: {len(target_data)}")
    print(f"Number of unique values: {target_data.nunique()}")
    print(f"Value counts:\n{target_data.value_counts()}")
    print(f"Number of NaN values: {target_data.isna().sum()}")


def run_statistical_analysis(
    data: pd.DataFrame, features: List[str], target: str, auto_correct=False
) -> pd.DataFrame:
    """
    Run statistical analysis on multiple features in parallel.

    Args:
        data: DataFrame with features and target.
        features: List of feature names to analyze.
        target: Name of target variable.
        auto_correct: If True and low cell counts for chi2 detected, switch
                      to simulated p-value.

    Returns:
        DataFrame of analysis results, sorted by corrected p-value.
    """
    diagnose_target(data, target)
    print("\nProceeding with feature analysis...")
    results = dict(
        Parallel(n_jobs=-1)(
            delayed(analyze_feature)(
                feature, data, target, auto_correct=auto_correct
            )
            for feature in tqdm(features, desc="Analyzing features")
        )
    )

    results = {k: v for k, v in results.items() if v is not None}

    features = list(results.keys())
    p_values = [results[feature]["p_value"] for feature in features]
    _, corrected_p_values, _, _ = multipletests(p_values, method="fdr_bh")

    for i, feature in enumerate(features):
        results[feature]["corrected_p_value"] = corrected_p_values[i]

    return pd.DataFrame.from_dict(results, orient="index").sort_values(
        "corrected_p_value"
    )
