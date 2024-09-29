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

import os
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pickle
from joblib import parallel_backend
import lightgbm as lgb
import numpy as np
import optuna
from optuna import Trial
from optuna.storages import RDBStorage
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
    fbeta_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FunctionTransformer
from sklearn.model_selection import cross_val_predict


from sklearn.pipeline import Pipeline


def evaluate_model(
    name: str,
    model: Union[Pipeline, Any],
    x: np.ndarray,
    y: np.ndarray,
    checkpoint_dir: str = "../models",
) -> Dict[str, Any]:
    """
    Evaluate a machine learning model using cross-validation and save checkpoints.

    This function performs the following steps:
    1. Loads a checkpoint if it exists
    2. Performs stratified k-fold cross-validation only if necessary
    3. Calculates performance metrics (precision, recall, F1-score, AUC-ROC)
    4. Saves checkpoint with results if new evaluation was performed

    Args:
        name (str): Name of the model being evaluated.
        model (Union[Pipeline, Any]): The machine learning model or pipeline to evaluate.
        x (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        checkpoint_dir (str): Directory to save/load model checkpoints.

    Returns:
        Dict[str, Any]: A dictionary containing the model name and performance metrics.
    """
    print(f"Evaluating {name}...")

    checkpoint = load_checkpoint(name, checkpoint_dir)
    if (
        checkpoint
        and isinstance(checkpoint, dict)
        and "y_pred" in checkpoint
        and "y_pred_proba" in checkpoint
    ):
        print(f"Resumed from checkpoint for model {name}.")
        model = checkpoint.get(
            "model", model
        )  # Use stored model if available, else use the provided one
        y_pred = checkpoint["y_pred"]
        y_pred_proba = checkpoint["y_pred_proba"]
    else:
        print(f"Performing cross-validation for {name}")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, x, y, cv=cv, method="predict")
        y_pred_proba = cross_val_predict(
            model, x, y, cv=cv, method="predict_proba"
        )[:, 1]

        save_checkpoint(
            {"model": model, "y_pred": y_pred, "y_pred_proba": y_pred_proba},
            name,
            checkpoint_dir,
        )
        print(f"Saved new checkpoint for {name}")

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_proba)

    print(f"{name} Cross-validation results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("=" * 60 + "\n")

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
    }


def save_checkpoint(
    data: Union[Dict[str, Any], Pipeline], name: str, directory: str
) -> None:
    """
    Save the model checkpoint and evaluation results to the specified directory.

    Args:
        data (Union[Dict[str, Any], Pipeline]): Dictionary containing model and evaluation results, or a Pipeline object.
        name (str): Name of the model, used to generate the filename.
        directory (str): Directory path where the checkpoint will be saved.

    Raises:
        IOError: If there's an issue writing the file.
    """
    os.makedirs(directory, exist_ok=True)
    filename = f"{name.lower().replace(' ', '_')}_checkpoint.pkl"
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved checkpoint: {filepath}")
        if isinstance(data, dict):
            print(f"Checkpoint contents: {list(data.keys())}")
        elif isinstance(data, Pipeline):
            print("Saved Pipeline object as checkpoint")
    except IOError as e:
        print(f"Error saving checkpoint: {e}")
        raise


def load_checkpoint(name: str, directory: str) -> Optional[Dict[str, Any]]:
    """
    Load the model checkpoint and evaluation results from the specified directory.

    Args:
        name (str): Name of the model, used to generate the filename.
        directory (str): Directory path where the checkpoint file is located.

    Returns:
        Optional[Dict[str, Any]]: The loaded checkpoint data if successful, None otherwise.

    Raises:
        IOError: If there's an issue reading the file.
    """
    filename = f"{name.lower().replace(' ', '_')}_checkpoint.pkl"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded checkpoint: {filepath}")
            return data
        except (IOError, pickle.UnpicklingError) as e:
            print(f"Error loading checkpoint: {e}")
            return None
    print(f"No checkpoint found at: {filepath}")
    return None


def save_progress(
    trial: Trial, name: str, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> None:
    """
    Save the evaluation progress directly to the Optuna trial's user attributes.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        name (str): Name of the model.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
    """
    trial.set_user_attr(
        "progress",
        {
            "model": name,
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
        },
    )


def load_progress(trial: Trial) -> Dict[str, Any]:
    """
    Load the evaluation progress from the Optuna trial's user attributes.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        Dict[str, Any]: The loaded progress data. An empty dictionary is returned
                       if no progress data is found.
    """
    try:
        progress_data = trial.user_attrs["progress"]
        print(f"Loaded progress from trial {trial.number}")
        return progress_data
    except KeyError:
        print(f"No progress found in trial {trial.number}")
        return {}


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
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Downscale numeric columns and encode categoricals based on df_train.

    This function optimizes memory usage by downcasting numeric columns to
    the smallest possible data type that can represent all values without
    loss of information. It also ensures consistent categorical encoding
    between train and test datasets. The target column, if specified and
    present, is ensured to be numeric.

    Args:
        df_train: Training DataFrame to be downscaled.
        df_test: Optional test DataFrame to be downscaled using the same
            rules as df_train.
        target_column: Optional name of the target column to ensure it's
            numeric.

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
        if col == target_column and col in df_train.columns:
            downscale_actions[col] = _ensure_numeric_target(df_train[col])
        elif pd.api.types.is_numeric_dtype(df_train[col]):
            downscale_actions[col] = _get_optimal_numeric_type(df_train[col])
        elif hasattr(df_train[col].dtype, "categories"):
            categorical_actions[col] = df_train[col].cat.categories

    df_train = df_train.astype(downscale_actions)

    if df_test is not None:
        # Only apply downscaling to columns that exist in df_test
        test_downscale_actions = {
            col: dtype
            for col, dtype in downscale_actions.items()
            if col in df_test.columns
        }
        df_test = df_test.astype(test_downscale_actions)

        for col, categories in categorical_actions.items():
            if col in df_test.columns:
                df_test[col] = pd.Categorical(
                    df_test[col], categories=categories
                ).astype("category")
        return df_train, df_test

    return df_train


def _ensure_numeric_target(series: pd.Series) -> np.dtype:
    """
    Ensure the target column is numeric and determine its optimal dtype.

    Args:
        series: The pandas Series representing the target column.

    Returns:
        The optimal numpy dtype for the target series.
    """
    if pd.api.types.is_categorical_dtype(series):
        series = series.cat.codes
    elif not pd.api.types.is_numeric_dtype(series):
        series = pd.factorize(series)[0]

    return _get_optimal_numeric_type(series)


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


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize feature names in a pandas DataFrame.

    Replaces any sequence of non-word characters with a single underscore
    and ensures that column names do not start or end with an underscore.

    Args:
        df: A pandas DataFrame whose column names need to be sanitized.

    Returns:
        A pandas DataFrame with sanitized column names.
    """

    def sanitize_column(col):
        # Replace non-word characters with underscore
        sanitized = re.sub(r"[^\w]+", "_", col)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        return sanitized

    sanitized_columns = {col: sanitize_column(col) for col in df.columns}
    return df.rename(columns=sanitized_columns)


sanitize_transformer: Callable = FunctionTransformer(sanitize_feature_names)


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists. Create it if it does not exist.

    Args:
        directory: The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def optimize_hyperparameters(
    x: pd.DataFrame,
    y,
    model_type,
    n_trials=100,
    cv=5,
    random_state=42,
    n_jobs=-1,
    checkpoint_dir="../models",
    study_name="hyperparameter_optimization",
    storage="sqlite:///data/optuna_study.db",
):
    """
    Perform hyperparameter optimization using Optuna for xGBoost or LightGBM models,
    focusing on maximizing the F2 score. Supports resuming from previous runs by using persistent storage.

    Args:
        x (pd.DataFrame): Features.
        y: Target.
        model_type (str): 'xgboost' or 'lightgbm'.
        n_trials (int): Number of trials to run.
        cv (int): Cross-validation folds.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs. If -1, all cores will be used.
        checkpoint_dir (str): Directory to save model checkpoints.
        study_name (str): Name of the Optuna study.
        storage (str): SQLite storage for saving study data.

    Returns:
        dict: The best hyperparameters found.
        float: The best F2 score achieved.
        object: The best model loaded from the checkpoint.
    """

    x_sanitized = sanitize_feature_names(x)

    def objective(trial: Trial):
        if model_type == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 1.0
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 10
                ),
                "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_uniform(
                    "colsample_bytree", 0.6, 1.0
                ),
                "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
                "scale_pos_weight": trial.suggest_loguniform(
                    "scale_pos_weight", 1, 100
                ),
                "random_state": random_state,
            }
            model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 1.0
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 1, 300
                ),
                "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_uniform(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                "reg_lambda": trial.suggest_loguniform(
                    "reg_lambda", 1e-8, 10.0
                ),
                "class_weight": "balanced",
                "random_state": random_state,
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(
                "model_type must be either 'xgboost' or 'lightgbm'"
            )

        f2_scorer = make_scorer(fbeta_score, beta=2)

        with parallel_backend("loky"):
            scores = cross_val_score(
                model, x_sanitized, y, cv=cv, scoring=f2_scorer, n_jobs=n_jobs
            )

        return scores.mean()

    os.makedirs(checkpoint_dir, exist_ok=True)
    storage_dir = os.path.dirname(storage.replace("sqlite:///", ""))
    if storage_dir:
        os.makedirs(storage_dir, exist_ok=True)

    storage_backend = RDBStorage(url=storage)
    try:
        study = optuna.load_study(
            study_name=study_name, storage=storage_backend
        )
        print(f"Loaded existing study '{study_name}'.")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_backend,
            direction="maximize",
            load_if_exists=True,
        )
        print(f"Created new study '{study_name}'.")

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_model = load_checkpoint(
        f"{model_type}_trial_{study.best_trial.number}", checkpoint_dir
    )

    return study.best_params, study.best_value, best_model
