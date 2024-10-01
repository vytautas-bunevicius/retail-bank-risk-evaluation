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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pickle
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
    fbeta_score,
)
from sklearn.pipeline import FunctionTransformer, Pipeline


def evaluate_model(
    name: str,
    model: Union[Pipeline, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    checkpoint_dir: str = "../models",
    is_tuned: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a machine learning model using a separate validation set and save checkpoints.

    This function performs the following steps:
    1. Loads a checkpoint if it exists
    2. Trains the model on the training data if necessary
    3. Makes predictions on the validation set
    4. Calculates performance metrics (precision, recall, F1-score, F2-score, AUC-ROC)
    5. Saves checkpoint with results if new evaluation was performed

    Args:
        name (str): Name of the model being evaluated.
        model (Union[Pipeline, Any]): The machine learning model or pipeline to evaluate.
        x_train (pd.DataFrame): Training feature data.
        y_train (np.ndarray): Training target data.
        x_val (pd.DataFrame): Validation feature data.
        y_val (np.ndarray): Validation target data.
        checkpoint_dir (str): Directory to save/load model checkpoints.
        is_tuned (bool): Flag indicating if the model is a result of hyperparameter tuning.

    Returns:
        Dict[str, Any]: A dictionary containing the model name and performance metrics.
    """
    print(f"Evaluating {'tuned ' if is_tuned else ''}{name}...")

    checkpoint = load_checkpoint(name, checkpoint_dir, is_tuned)
    if checkpoint and isinstance(checkpoint, dict) and "model" in checkpoint:
        print(
            f"Resumed from checkpoint for {'tuned ' if is_tuned else ''}model {name}."
        )
        model = checkpoint["model"]
    else:
        print(f"Training {'tuned ' if is_tuned else ''}model {name}")
        model.fit(x_train, y_train)
        save_checkpoint({"model": model}, name, checkpoint_dir, is_tuned)
        print(f"Saved new checkpoint for {'tuned ' if is_tuned else ''}{name}")

    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)[:, 1]

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    f2 = fbeta_score(y_val, y_pred, beta=2)
    auc_roc = roc_auc_score(y_val, y_pred_proba)

    print(f"{'Tuned ' if is_tuned else ''}{name} Validation results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("=" * 60 + "\n")

    return {
        "model": f"{'tuned_' if is_tuned else ''}{name}",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "auc_roc": auc_roc,
    }


def save_checkpoint(
    data: Union[Dict[str, Any], Pipeline],
    name: str,
    directory: str,
    is_tuned: bool = False,
) -> None:
    """
    Save the model checkpoint and evaluation results to the specified directory.

    Args:
        data (Union[Dict[str, Any], Pipeline]): Dictionary containing model and
                                                evaluation results, or a Pipeline object.
        name (str): Name of the model, used to generate the filename.
        directory (str): Directory path where the checkpoint will be saved.
        is_tuned (bool): Flag indicating if the model is a result of hyperparameter tuning.

    Raises:
        IOError: If there's an issue writing the file.
    """
    os.makedirs(directory, exist_ok=True)
    prefix = "tuned_" if is_tuned else ""
    filename = f"{prefix}{name.lower().replace(' ', '_')}_checkpoint.pkl"
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {'tuned ' if is_tuned else ''}checkpoint: {filepath}")
        if isinstance(data, dict):
            print(f"Checkpoint contents: {list(data.keys())}")
        elif isinstance(data, Pipeline):
            print(
                f"Saved {'tuned ' if is_tuned else ''}Pipeline object as checkpoint"
            )
    except IOError as e:
        print(f"Error saving checkpoint: {e}")
        raise


def load_checkpoint(
    name: str, directory: str, is_tuned: bool = False
) -> Optional[Union[Dict[str, Any], Pipeline]]:
    """
    Load the model checkpoint and evaluation results from the specified directory.

    Args:
        name (str): Name of the model, used to generate the filename.
        directory (str): Directory path where the checkpoint file is located.
        is_tuned (bool): Flag indicating if the model to load is a result of hyperparameter tuning.

    Returns:
        Optional[Union[Dict[str, Any], Pipeline]]: The loaded checkpoint data
        if successful, None otherwise.

    Raises:
        IOError: If there's an issue reading the file.
    """
    prefix = "tuned_" if is_tuned else ""
    filename = f"{prefix}{name.lower().replace(' ', '_')}_checkpoint.pkl"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(
                f"Loaded {'tuned ' if is_tuned else ''}checkpoint: {filepath}"
            )
            return data
        except (IOError, pickle.UnpicklingError) as e:
            print(f"Error loading checkpoint: {e}")
            return None
    print(f"No {'tuned ' if is_tuned else ''}checkpoint found at: {filepath}")
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

    Replaces non-word characters with underscores and removes leading/trailing
    underscores from column names.

    Args:
        df: A pandas DataFrame whose column names need to be sanitized.

    Returns:
        A pandas DataFrame with sanitized column names.
    """
    sanitized_columns = df.columns.str.replace(
        r"\W+", "_", regex=True
    ).str.strip("_")
    return df.rename(columns=dict(zip(df.columns, sanitized_columns)))


sanitize_transformer: Callable = FunctionTransformer(sanitize_feature_names)
sanitizer = FunctionTransformer(sanitize_feature_names)


def validate_data(x: pd.DataFrame, y: pd.Series, name: str) -> None:
    """
    Validate input data for machine learning pipeline.

    Args:
        x: Feature DataFrame.
        y: Target Series.
        name: Name of the dataset for logging purposes.

    Raises:
        AssertionError: If data validation fails.
    """
    assert isinstance(x, pd.DataFrame), f"{name} x must be a pandas DataFrame"
    assert isinstance(y, pd.Series), f"{name} y must be a pandas Series"
    assert (
        x.shape[0] == y.shape[0]
    ), f"{name} x and y must have the same number of samples"
    assert not x.isnull().any().any(), f"{name} x contains null values"
    assert not y.isnull().any(), f"{name} y contains null values"
    print(
        f"{name} data validation passed. x shape: {x.shape}, y shape: {y.shape}"
    )


def get_model_and_params(
    model_type: str, trial: Trial, random_state: int, n_jobs: int
) -> Tuple[Any, Dict[str, Any]]:
    """
    Get model class and hyperparameter search space.

    Args:
        model_type: Type of model ('xgboost' or 'lightgbm').
        trial: Optuna trial object.
        random_state: Random state for reproducibility.
        n_jobs: Number of parallel jobs.

    Returns:
        Tuple containing model class and hyperparameter dictionary.

    Raises:
        ValueError: If model_type is neither 'xgboost' nor 'lightgbm'.
    """
    if model_type == "xgboost":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-3, 1.0
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
            "scale_pos_weight": trial.suggest_loguniform(
                "scale_pos_weight", 1, 100
            ),
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": n_jobs,
        }
        return xgb.XGBClassifier, params
    elif model_type == "lightgbm":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-3, 1.0
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        return lgb.LGBMClassifier, params
    else:
        raise ValueError("model_type must be either 'xgboost' or 'lightgbm'")


def train_and_evaluate(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    feature_selection_threshold: float,
) -> Tuple[float, List[str]]:
    """
    Train model, perform feature selection, and evaluate on validation set.

    Args:
        model: Machine learning model object.
        x_train: Training feature DataFrame.
        y_train: Training target Series.
        x_val: Validation feature DataFrame.
        y_val: Validation target Series.
        feature_selection_threshold: Threshold for feature selection.

    Returns:
        Tuple containing F2 score and list of selected features.
    """
    model.fit(x_train, y_train)

    importances = model.feature_importances_
    feature_imp = pd.DataFrame(
        {"Value": importances, "Feature": x_train.columns}
    )
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)
    num_top_features = max(
        1, int(len(feature_imp) * feature_selection_threshold)
    )
    top_features = feature_imp.head(num_top_features)["Feature"].tolist()

    x_train_selected = x_train[top_features]
    x_val_selected = x_val[top_features]
    model.fit(x_train_selected, y_train)

    y_pred = model.predict(x_val_selected)
    f2 = fbeta_score(y_val, y_pred, beta=2)
    return f2, top_features


def objective(
    trial: Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    random_state: int,
    n_jobs: int,
    feature_selection_threshold: float,
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object.
        x_train: Training feature DataFrame.
        y_train: Training target Series.
        x_val: Validation feature DataFrame.
        y_val: Validation target Series.
        model_type: Type of model ('xgboost' or 'lightgbm').
        random_state: Random state for reproducibility.
        n_jobs: Number of parallel jobs.
        feature_selection_threshold: Threshold for feature selection.

    Returns:
        F2 score.

    Raises:
        Exception: If an error occurs during the trial.
    """
    model_class, params = get_model_and_params(
        model_type, trial, random_state, n_jobs
    )
    model = model_class(**params)

    try:
        f2, _ = train_and_evaluate(
            model, x_train, y_train, x_val, y_val, feature_selection_threshold
        )
        return f2
    except Exception as e:
        print(f"Error in trial: {str(e)}")
        raise


def optimize_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    n_trials: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    checkpoint_dir: str = "../models",
    study_name: str = "hyperparameter_optimization",
    storage: str = "sqlite:///data/optuna_study.db",
    feature_selection_threshold: float = 1,
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        x_train: Training feature DataFrame.
        y_train: Training target Series.
        x_val: Validation feature DataFrame.
        y_val: Validation target Series.
        x_test: Test feature DataFrame.
        y_test: Test target Series.
        model_type: Type of model ('xgboost' or 'lightgbm').
        n_trials: Number of optimization trials.
        random_state: Random state for reproducibility.
        n_jobs: Number of parallel jobs.
        checkpoint_dir: Directory to save model checkpoints.
        study_name: Name of the Optuna study.
        storage: Storage URL for the Optuna study.
        feature_selection_threshold: Threshold for feature selection.

    Returns:
        Dictionary containing optimization results and best model performance.
    """
    print(f"Optimizing hyperparameters for {model_type}...")

    validate_data(x_train, y_train, "Training")
    validate_data(x_val, y_val, "Validation")
    validate_data(x_test, y_test, "Test")

    storage_backend = RDBStorage(url=storage)

    try:
        study = optuna.load_study(
            study_name=study_name, storage=storage_backend
        )
        print(
            f"Loaded existing study '{study_name}' with {len(study.trials)} trials."
        )
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_backend,
            direction="maximize",
            load_if_exists=True,
        )
        print(f"Created new study '{study_name}'.")

    completed_trials = len(study.trials)
    print(f"Number of completed trials for {model_type}: {completed_trials}")

    remaining_trials = n_trials - completed_trials
    if remaining_trials > 0:
        print(
            f"Running {remaining_trials} more trials for {model_type} "
            f"to reach {n_trials} in total."
        )

        study.optimize(
            lambda trial: objective(
                trial,
                x_train,
                y_train,
                x_val,
                y_val,
                model_type,
                random_state,
                n_jobs,
                feature_selection_threshold,
            ),
            n_trials=remaining_trials,
            n_jobs=1,
        )
    else:
        print(f"Study has already completed {n_trials} or more trials.")

    best_params = study.best_params
    model_class, _ = get_model_and_params(
        model_type, study.best_trial, random_state, n_jobs
    )
    best_model = model_class(**best_params)

    x_train_full = pd.concat([x_train, x_val])
    y_train_full = pd.concat([y_train, y_val])

    _, top_features = train_and_evaluate(
        best_model,
        x_train_full,
        y_train_full,
        x_test,
        y_test,
        feature_selection_threshold,
    )

    x_test_selected = x_test[top_features]
    y_pred = best_model.predict(x_test_selected)
    y_pred_proba = best_model.predict_proba(x_test_selected)[:, 1]

    results = {
        "model": f"tuned_{model_type}",
        "best_params": best_params,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "f2_score": fbeta_score(y_test, y_pred, beta=2),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "selected_features": top_features,
    }

    print(f"{model_type.capitalize()} Best Model Results (Test Set):")
    for key, value in results.items():
        if key not in ["model", "best_params", "selected_features"]:
            print(f"{key}: {value:.4f}")
    print(f"Number of selected features: {len(top_features)}")

    save_checkpoint(
        {
            "model": best_model,
            "params": best_params,
            "selected_features": top_features,
        },
        results["model"],
        checkpoint_dir,
        is_tuned=True,
    )
    print(f"Saved best tuned {model_type} model checkpoint.")

    return results
