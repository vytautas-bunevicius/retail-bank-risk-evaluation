"""
This module contains a collection of functions for visualizing and analyzing
machine learning model performance, including SHAP values, model comparison
metrics, and confusion matrices.

The visualizations are created using Plotly and are designed to be informative,
aesthetically pleasing, and easily interpretable. The module also supports
saving these visualizations as images.

Key functionalities include:
- `shap_summary_plot`: Creates a bar plot to display SHAP feature importance.
- `shap_force_plot`: Generates a waterfall plot to visualize individual SHAP
  values and their impact on predictions.
- `plot_model_performance`: Plots and compares performance metrics across
  different models using a grouped bar chart.
- `plot_combined_confusion_matrices`: Creates confusion matrices for multiple
  models, visualizing their classification performance.

Constants defined:
- `BACKGROUND_COLOR`: The background color used for all plots.
- `PRIMARY_COLORS`: A palette of primary colors used for the main elements of
  the visualizations.
- `PLOT_COLORS`: A subset of colors used specifically for plotting.
- `SECONDARY_COLORS`: A palette of secondary colors for additional elements.
- `ALL_COLORS`: A combination of primary and secondary colors for use across
  different plots.

This module is intended for use in data analysis workflows where model
interpretability and performance visualization are essential. The plots
generated can be displayed interactively or saved for reporting purposes.
"""

from typing import List, Optional, Dict, Any
import math

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import learning_curve

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BACKGROUND_COLOR = "#EEECE2"
PRIMARY_COLORS = ["#CC7B5C", "#D4A27F", "#EBDBBC", "#9C8AA5"]
PLOT_COLORS = ["#91A694", "#9C8AA5", "#CC7B5C"]
SECONDARY_COLORS = [
    "#91A694",
    "#8B9BAE",
    "#666663",
    "#BFBFBA",
    "#E5E4DF",
    "#F0F0EB",
    "#FAFAF7",
]
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def shap_summary_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Creates a bar plot of SHAP feature importance.

    Args:
        shap_values (np.ndarray): The SHAP values for each feature.
        feature_names (List[str]): The names of the features.
        save_path (Optional[str]): The file path to save the plot image. Defaults to None.

    Returns:
        None: The function displays the plot and optionally saves it.
    """
    shap_mean = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": shap_mean}
    )
    feature_importance = feature_importance.sort_values(
        "importance", ascending=True
    )

    fig = px.bar(
        feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        title="SHAP Feature Importance",
        labels={"importance": "mean(|SHAP value|)", "feature": "Feature"},
        color="importance",
        color_continuous_scale=PRIMARY_COLORS,
    )

    fig.update_layout(
        height=1200,
        width=1200,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font={"family": "Styrene A", "size": 12, "color": "#191919"},
        title={
            "text": "SHAP Feature Importance",
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": "Styrene B", "size": 20, "color": "#191919"},
        },
    )

    if save_path:
        fig.write_image(save_path)


def shap_force_plot(
    shap_data: Dict[str, Any],
    explainer: Any,
    idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Generates a waterfall plot to visualize individual SHAP values and their
    impact on predictions.

    Args:
        shap_data (Dict[str, Any]): A dictionary containing 'shap_values',
        'x_data', and 'feature_names'.
        explainer (Any): The SHAP explainer object.
        idx (int, optional): The index of the instance to plot. Defaults to 0.
        save_path (Optional[str]): The file path to save the plot as an image.
        Defaults to None.

    Returns:
        None: This function does not return anything. It plots the SHAP values
        using Plotly.
    """
    shap_values = shap_data["shap_values"]
    x_data = shap_data["x_data"]
    feature_names = shap_data["feature_names"]

    features = pd.DataFrame(
        {
            "feature": feature_names,
            "value": x_data[idx],
            "shap": shap_values[idx],
        }
    )
    features = features.sort_values("shap", key=abs, ascending=False)

    base_value = getattr(explainer, "expected_value", None)
    if base_value is None:
        raise AttributeError(
            "The explainer object does not have an 'expected_value' attribute."
        )

    fig = go.Figure(
        go.Waterfall(
            name="20",
            orientation="h",
            measure=["relative"] * len(features) + ["total"],
            x=list(features.loc[:, "shap"]) + [base_value],
            textposition="outside",
            text=[
                f"{feat}: {val:.2f}"
                for _, row in features.iterrows()
                for feat, val in zip(row["feature"], row["value"])
            ]
            + ["Base value"],
            y=list(features.loc[:, "feature"]) + ["Base value"],
            connector={"line": {"color": "#666663"}},
            decreasing={"marker": {"color": PRIMARY_COLORS[0]}},
            increasing={"marker": {"color": PRIMARY_COLORS[1]}},
        )
    )

    fig.update_layout(
        height=800,
        width=800,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font={"family": "Styrene A", "size": 12, "color": "#191919"},
        title={"font": {"family": "Styrene B", "size": 20, "color": "#191919"}},
    )
    if save_path:
        fig.write_image(save_path)


def plot_model_performance(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plots and optionally saves a bar chart of model performance metrics with legend on the right.

    Args:
        results: A dictionary with model names as keys and dicts of performance metrics as values.
        metrics: List of performance metrics to plot (e.g., 'Accuracy', 'Precision').
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    model_names = list(results.keys())
    data = {
        metric: [results[name][metric] for name in model_names]
        for metric in metrics
    }

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=data[metric],
                name=metric,
                marker_color=ALL_COLORS[i % len(ALL_COLORS)],
                text=[f"{value:.2f}" for value in data[metric]],
                textposition="auto",
            )
        )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    fig.update_layout(
        barmode="group",
        title={
            "text": "Comparison of Model Performance Metrics",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": "Styrene B", "size": 24, "color": "#191919"},
        },
        xaxis_title="Model",
        yaxis_title="Value",
        legend_title="Metrics",
        font={**axis_font, "size": 14},
        height=500,
        width=1200,
        template="plotly_white",
        legend={"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
    )

    fig.update_yaxes(
        range=[0, 1], showgrid=True, gridwidth=1, gridcolor="LightGrey"
    )
    fig.update_xaxes(tickangle=-45, tickfont={**axis_font, "size": 12})

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_combined_confusion_matrices(
    results: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    y_pred_dict: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots combined confusion matrices for multiple models.

    Args:
        results (Dict[str, Dict[str, float]]): A dictionary containing the results
        of each model. The keys are the model names, and the values are
        dictionaries containing the model's performance metrics.
        y_test (np.ndarray): The true labels of the test data.
        y_pred_dict (Dict[str, np.ndarray]): A dictionary containing the predicted
        labels for each model. The keys are the model names, and the values are
        the predicted labels.
        labels (Optional[List[str]], optional): A list of labels for the classes.
        Defaults to None.
        save_path (Optional[str], optional): The file path to save the plot as an
        image. Defaults to None.

    Returns:
        None: This function does not return anything. It plots the confusion
        matrices using Plotly.
    """

    n_models = len(results)

    # Determine the grid layout
    if n_models <= 2:
        rows, cols = 1, 2
    elif n_models <= 4:
        rows, cols = 2, 2
    else:
        rows = math.ceil(n_models / 2)
        cols = 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(results.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, (name, _) in enumerate(results.items()):
        row = i // cols + 1
        col = i % cols + 1

        cm = confusion_matrix(y_test, y_pred_dict[name])
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        text = [
            [
                f"TN: {cm[0][0]}<br>({cm_percent[0][0]:.1f}%)",
                f"FP: {cm[0][1]}<br>({cm_percent[0][1]:.1f}%)",
            ],
            [
                f"FN: {cm[1][0]}<br>({cm_percent[1][0]:.1f}%)",
                f"TP: {cm[1][1]}<br>({cm_percent[1][1]:.1f}%)",
            ],
        ]

        colorscale = [
            [0, PRIMARY_COLORS[2]],
            [0.33, PRIMARY_COLORS[1]],
            [0.66, PRIMARY_COLORS[1]],
            [1, PRIMARY_COLORS[0]],
        ]

        heatmap = go.Heatmap(
            z=cm,
            x=labels or ["Non-Defaulter", "Defaulter"],
            y=labels or ["Non-Defaulter", "Defaulter"],
            hoverongaps=False,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
        )

        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(
            title_text="Predicted",
            row=row,
            col=col,
            tickfont={**axis_font, "size": 10},
            title_standoff=25,
        )
        fig.update_yaxes(
            title_text="Actual",
            row=row,
            col=col,
            tickfont={**axis_font, "size": 10},
            title_standoff=25,
        )

    height = max(400 * rows, 800)  # Ensure minimum height of 800
    width = 1200

    fig.update_layout(
        title_text="Confusion Matrices for All Models",
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 24, "color": "#191919"},
        height=height,
        width=width,
        showlegend=False,
        font={**axis_font, "size": 12},
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        margin={"t": 100, "b": 50, "l": 50, "r": 50},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot and optionally save the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: Array of true labels.
        y_pred_proba: Array of predicted probabilities.
        save_path: Optional path to save the plot image.

    Returns:
        go.Figure: The plotly Figure object containing the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC curve",
            line={"color": PRIMARY_COLORS[0]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": PRIMARY_COLORS[1]},
        )
    )
    fig.update_layout(
        title={
            "text": "Receiver Operating Characteristic (ROC) Curve",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        width=1200,
        height=500,
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and optionally save the Precision-Recall curve.

    Args:
        y_true: Array of true labels.
        y_pred_proba: Array of predicted probabilities.
        save_path: Optional path to save the plot image.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="PR curve",
            line={"color": PRIMARY_COLORS[0]},
        )
    )
    fig.update_layout(
        title={
            "text": "Precision-Recall Curve",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Recall",
        yaxis_title="Precision",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        width=1200,
        height=500,
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots and optionally saves the confusion matrix.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        labels (Optional[List[str]], optional): A list of labels for the classes.
        Defaults to None.
        save_path (Optional[str], optional): The file path to save the plot as an
        image. Defaults to None.

    Returns:
        None: This function does not return anything. It plots the confusion
        matrix using Plotly.

    Raises:
        None: This function does not raise any exceptions.
    """

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    text = [
        [
            f"TN: {cm[0][0]}<br>({cm_percent[0][0]:.1f}%)",
            f"FP: {cm[0][1]}<br>({cm_percent[0][1]:.1f}%)",
        ],
        [
            f"FN: {cm[1][0]}<br>({cm_percent[1][0]:.1f}%)",
            f"TP: {cm[1][1]}<br>({cm_percent[1][1]:.1f}%)",
        ],
    ]

    colorscale = [
        [0, PRIMARY_COLORS[2]],
        [0.33, PRIMARY_COLORS[1]],
        [0.66, PRIMARY_COLORS[1]],
        [1, PRIMARY_COLORS[0]],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels or ["Not Transported", "Transported"],
            y=labels or ["Not Transported", "Transported"],
            hoverongaps=False,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
        )
    )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    fig.update_layout(
        title_text="Confusion Matrix",
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 24, "color": "#191919"},
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis={"tickfont": {**axis_font, "size": 12}},
        yaxis={"tickfont": {**axis_font, "size": 12}},
        height=500,
        width=1200,
        showlegend=False,
        font={**axis_font, "size": 12},
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        margin={"t": 100, "b": 50, "l": 50, "r": 50},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_learning_curve(
    estimator,
    features: np.ndarray,
    target: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and optionally save the learning curve for a given estimator.

    Args:
        estimator: The machine learning model to evaluate.
        features: Feature matrix.
        target: Target vector.
        cv: Number of cross-validation folds.
        n_jobs: Number of jobs to run in parallel.
        train_sizes: Array of training set sizes to evaluate.
        save_path: Optional path to save the plot image.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        features,
        target,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode="lines+markers",
            name="Training score",
            line={"color": PRIMARY_COLORS[0]},
            error_y={
                "type": "data",
                "array": train_scores_std,
                "visible": True,
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode="lines+markers",
            name="Cross-validation score",
            line={"color": PRIMARY_COLORS[1]},
            error_y={"type": "data", "array": test_scores_std, "visible": True},
        )
    )
    fig.update_layout(
        title={
            "text": "Learning Curve",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Training examples",
        yaxis_title="Score",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        width=1200,
        height=500,
    )

    if save_path:
        fig.write_image(save_path)

    return fig
