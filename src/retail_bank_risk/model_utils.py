"""
This module provides functions for evaluating machine learning models and extracting feature importances.

Functions included:
- `evaluate_model`: Evaluates the performance of a trained model using various metrics, including optional threshold adjustment.
- `extract_feature_importances`: Extracts feature importances using permutation importance for models that do not provide them directly.

Key Features:
- Evaluation metrics include ROC AUC, PR AUC, F1 Score, Precision, Recall, and Balanced Accuracy.
- The `evaluate_model` function allows for adjusting the decision threshold based on a target recall.
- The `extract_feature_importances` function uses permutation importance for models that lack direct feature importance attributes.

This module is intended for use in machine learning workflows to assess model performance and interpret feature importance.
"""

from typing import Dict, Union

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
        _, recalls, thresholds = precision_recall_curve(true_labels, y_pred_proba)
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
    print(f"Precision: {precision_score(true_labels, y_pred, zero_division=1):.4f}")
    print(f"Recall: {recall_score(true_labels, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(true_labels, y_pred):.4f}")

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
    Extract feature importances using permutation importance for models that do not directly provide them.

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
