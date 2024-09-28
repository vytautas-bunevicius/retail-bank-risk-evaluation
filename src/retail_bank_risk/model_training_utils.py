"""
This module provides functions for evaluating machine learning models and
extracting feature importances.

Functions included:
- `evaluate_model`: Evaluates the performance of a trained model using various
  metrics, including optional threshold adjustment.
- `extract_feature_importances`: Extracts feature importances using
  permutation importance for models that do not provide them directly.

Key Features:
- Evaluation metrics include ROC AUC, PR AUC, F1 Score, Precision, Recall,
  and Balanced Accuracy.
- The `evaluate_model` function allows for adjusting the decision threshold
  based on a target recall.
- The `extract_feature_importances` function uses permutation importance for
  models that lack direct feature importance attributes.

This module is intended for use in machine learning workflows to assess model
performance and interpret feature importance.
"""

from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    features: np.ndarray,
    true_labels: np.ndarray,
    dataset_name: str = None,
    threshold: float = None,
    target_recall: float = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate a model's performance with optional threshold adjustment.

    Args:
        model: The trained model to evaluate.
        features (np.ndarray): Feature data.
        true_labels (np.ndarray): True labels.
        dataset_name (str, optional): Name of the dataset for display purposes.
        threshold (float, optional): Custom threshold for classification.
        target_recall (float, optional): Target recall for threshold adjustment.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Dictionary containing various performance metrics.
    """
    y_pred_proba = model.predict_proba(features)[:, 1]

    if target_recall is not None:
        _, recalls, thresholds = precision_recall_curve(
            true_labels, y_pred_proba
        )
        idx = np.argmin(np.abs(recalls - target_recall))
        threshold = thresholds[idx]
        print(f"Adjusted threshold: {threshold:.4f}")

    if threshold is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(features)

    if dataset_name:
        print(f"\nResults on {dataset_name} set:")

    print(classification_report(true_labels, y_pred, zero_division=1))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, y_pred))
    print(f"ROC AUC: {roc_auc_score(true_labels, y_pred_proba):.4f}")
    print(f"PR AUC: {average_precision_score(true_labels, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(true_labels, y_pred, zero_division=1):.4f}")
    print(
        f"Precision: {precision_score(true_labels, y_pred, zero_division=1):.4f}"
    )
    print(f"Recall: {recall_score(true_labels, y_pred):.4f}")
    print(
        f"Balanced Accuracy: {balanced_accuracy_score(true_labels, y_pred):.4f}"
    )

    return {
        "roc_auc": roc_auc_score(true_labels, y_pred_proba),
        "pr_auc": average_precision_score(true_labels, y_pred_proba),
        "f1": f1_score(true_labels, y_pred, zero_division=1),
        "precision": precision_score(true_labels, y_pred, zero_division=1),
        "recall": recall_score(true_labels, y_pred),
        "balanced_accuracy": balanced_accuracy_score(true_labels, y_pred),
        "threshold": threshold if threshold is not None else 0.5,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def extract_feature_importances(
    model, feature_data: pd.DataFrame, target_data: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Extract feature importances using permutation importance for models that do
    not directly provide them.

    Args:
        model: Trained model.
        feature_data (pd.DataFrame): Feature data.
        target_data (Union[pd.Series, np.ndarray]): Target data.

    Returns:
        np.ndarray: Array of feature importances.
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    else:
        permutation_import = permutation_importance(
            model, feature_data, target_data, n_repeats=30, random_state=42
        )
        return permutation_import.importances_mean


def downscale_dtypes(
    df_train: pd.DataFrame, df_test: pd.DataFrame = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Downscale numeric columns and encode categoricals based on df_train.

    This function optimizes memory usage by downcasting numeric columns to
    the smallest possible data type that can represent all values without
    loss of information. It also ensures consistent categorical encoding
    between train and test datasets.

    Args:
        df_train: Training DataFrame to be downscaled.
        df_test: Optional test DataFrame to be downscaled using the same
            rules as df_train.

    Returns:
        If df_test is None:
            A downscaled version of df_train.
        If df_test is provided:
            A tuple containing downscaled versions of (df_train, df_test).

    Raises:
        ValueError: If df_train is empty or contains no columns.
    """
    if df_train.empty or df_train.columns.empty:
        raise ValueError("df_train must not be empty and contain columns.")

    downscale_actions = {}
    categorical_actions = {}

    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            downscale_actions[col] = _get_optimal_numeric_type(df_train[col])
        elif hasattr(df_train[col].dtype, "categories"):
            categorical_actions[col] = df_train[col].cat.categories

    df_train = df_train.astype(downscale_actions)

    if df_test is not None:
        df_test = df_test.astype(downscale_actions)
        for col, categories in categorical_actions.items():
            if col in df_test.columns:
                df_test[col] = pd.Categorical(
                    df_test[col], categories=categories
                ).astype("category")
        return df_train, df_test

    return df_train


def _get_optimal_numeric_type(series: pd.Series) -> np.dtype:
    """
    Determine the optimal numeric dtype for a given series.

    Args:
        series: The pandas Series to analyze.

    Returns:
        The optimal numpy dtype for the series.
    """
    if pd.api.types.is_integer_dtype(series):
        return _get_optimal_integer_type(series)
    elif pd.api.types.is_float_dtype(series):
        return _get_optimal_float_type(series)
    return series.dtype


def _get_optimal_integer_type(series: pd.Series) -> np.dtype:
    """
    Determine the optimal integer dtype for a given series.

    Args:
        series: The pandas Series to analyze.

    Returns:
        The optimal numpy integer dtype for the series.
    """
    min_val, max_val = series.min(), series.max()
    if min_val >= 0:
        if max_val < np.iinfo(np.uint8).max:
            return np.uint8
        elif max_val < np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val < np.iinfo(np.uint32).max:
            return np.uint32
    else:
        if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
            return np.int8
        elif (
            min_val > np.iinfo(np.int16).min
            and max_val < np.iinfo(np.int16).max
        ):
            return np.int16
        elif (
            min_val > np.iinfo(np.int32).min
            and max_val < np.iinfo(np.int32).max
        ):
            return np.int32
    return series.dtype


def _get_optimal_float_type(series: pd.Series) -> np.dtype:
    """
    Determine the optimal float dtype for a given series.

    Args:
        series: The pandas Series to analyze.

    Returns:
        The optimal numpy float dtype for the series.
    """
    if (
        series.min() > np.finfo(np.float32).min
        and series.max() < np.finfo(np.float32).max
    ):
        return np.float32
    return series.dtype
