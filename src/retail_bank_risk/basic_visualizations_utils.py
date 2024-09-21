"""
This module provides a set of functions for creating and displaying various types
of plots to visualize data distributions, feature importances, and correlations
using Plotly.

Key functionalities include:
- Plotting histograms, bar charts, and boxplots for specified features.
- Creating correlation matrices for numerical features.
- Visualizing feature importances across different models.
- Comparing data distributions before and after handling missing values.
- Displaying categorical feature distributions by target variables.

Constants:
- `BACKGROUND_COLOR`: The background color used in all plots.
- `PRIMARY_COLORS`: A list of primary colors used for the main elements of
  the plots.
- `PLOT_COLORS`: A subset of colors used for histogram and bar plot elements.
- `SECONDARY_COLORS`: A set of secondary colors used for additional elements
  in the plots.
- `ALL_COLORS`: A combination of primary and secondary colors for use across
  different plots.

Functions:
- `plot_combined_histograms`: Plots histograms for multiple features in a
  single figure.
- `plot_combined_bar_charts`: Plots bar charts for multiple categorical features.
- `plot_combined_boxplots`: Plots boxplots for multiple numerical features.
- `plot_correlation_matrix`: Plots a correlation matrix for the specified
  numerical features.
- `plot_feature_importances`: Plots feature importances across different models.
- `plot_distribution_comparison`: Compares feature distributions before and
  after handling missing values.
- `plot_categorical_features_by_target`: Plots the distribution of categorical
  features grouped by a target variable.
- `plot_numeric_distributions`: Plots numeric distributions of features grouped
  by a binary target variable.
- `plot_single_bar_chart`: Plots a percentage bar chart for a single categorical
  feature.

This module is intended for use in data exploration and analysis workflows,
helping to gain insights into data structure, feature importance, and potential
relationships between features.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def plot_combined_histograms(
    df: pd.DataFrame,
    features: List[str],
    nbins: int = 40,
    save_path: Optional[str] = None,
) -> None:
    """Plots combined histograms for specified features in the DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of feature names to plot histograms for.
        nbins: Number of bins for each histogram. Defaults to 40.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    title = f"Distribution of {', '.join(features)}"
    rows, cols = 1, len(features)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
            ),
            row=1,
            col=i + 1,
        )

        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_standoff=25,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 80},
        font={**axis_font, "size": 12},
    )

    if save_path:
        fig.write_image(save_path)


def plot_combined_bar_charts(
    df: pd.DataFrame,
    features: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plots combined bar charts for specified categorical features in the DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of categorical feature names to plot bar charts for.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    title = f"Distribution of {', '.join(features)}"
    rows, cols = 1, len(features)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, feature in enumerate(features):
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, "count"]

        fig.add_trace(
            go.Bar(
                x=value_counts[feature],
                y=value_counts["count"],
                name=feature,
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
            ),
            row=1,
            col=i + 1,
        )

        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_standoff=25,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            showticklabels=True,
        )

        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 150},
        font={**axis_font, "size": 12},
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_combined_boxplots(
    df: pd.DataFrame, features: List[str], save_path: Optional[str] = None
) -> None:
    """Plots combined boxplots for specified numerical features in the DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of numerical feature names to plot boxplots for.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    title = f"Boxplots of {', '.join(features)}"
    rows, cols = 1, len(features)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Box(
                y=df[feature],
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
                boxmean="sd",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            title_text="Value",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )
        fig.update_xaxes(
            tickvals=[0],
            ticktext=[feature],
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            showticklabels=True,
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 150},
        font={**axis_font, "size": 12},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_features: List[str], save_path: str = None
) -> None:
    """
    Plots the correlation matrix of the specified numerical features in the
    DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_features (List[str]): List of numerical features to include in
        the correlation matrix.
        save_path (str): Path to save the image file (optional).
    """
    numerical_df = df[numerical_features]
    correlation_matrix = numerical_df.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale=PRIMARY_COLORS,
        title="Correlation Matrix",
    )

    fig.update_layout(
        title={
            "text": "Correlation Matrix",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        title_font=dict(size=24),
        template="plotly_white",
        height=800,
        width=800,
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis=dict(
            tickangle=-45, title_font=dict(size=18), tickfont=dict(size=14)
        ),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    )

    if save_path:
        fig.write_image(save_path)


def plot_feature_importances(
    feature_importances: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots and optionally saves a bar chart of feature importances across
    different models.

    Args:
        feature_importances: A dictionary with model names as keys and dicts of
        feature importances as values.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    fig = go.Figure()

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, (name, importances) in enumerate(feature_importances.items()):
        fig.add_trace(
            go.Bar(
                x=list(importances.keys()),
                y=list(importances.values()),
                name=name,
                marker_color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                text=[f"{value:.3f}" for value in importances.values()],
                textposition="auto",
            )
        )

    fig.update_layout(
        title={
            "text": "Feature Importances Across Models",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": "Styrene B", "size": 24, "color": "#191919"},
        },
        xaxis_title="Features",
        yaxis_title="Importance",
        barmode="group",
        template="plotly_white",
        legend_title="Models",
        font={**axis_font, "size": 14},
        height=600,
        width=1200,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
    )

    fig.update_xaxes(tickangle=-45, tickfont={**axis_font, "size": 12})
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont={**axis_font, "size": 12},
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_distribution_comparison(
    df_before,
    df_after,
    features,
    title="Distribution Comparison",
    save_path=None,
):
    """
    Plots the distribution of specified features before and after handling missing values.

    Args:
        df_before (pd.DataFrame): DataFrame before handling missing values.
        df_after (pd.DataFrame): DataFrame after handling missing values.
        features (list): List of feature names to plot.
        title (str): The title of the plot.

    Returns:
        None. Displays the plot.
    """
    n_features = len(features)
    fig = make_subplots(
        rows=n_features,
        cols=2,
        subplot_titles=[f"{feature} - Before" for feature in features]
        + [f"{feature} - After" for feature in features],
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df_before[feature],
                name="Before",
                marker_color=PRIMARY_COLORS[0],
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=df_after[feature],
                name="After",
                marker_color=PRIMARY_COLORS[1],
            ),
            row=i + 1,
            col=2,
        )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24, "color": "#191919", "family": "Styrene B"},
        },
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font={"family": "Styrene A", "size": 14, "color": "#191919"},
        height=400 * n_features,
        width=1200,
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_categorical_features_by_target(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of specified categorical features grouped by a target
    variable. This function creates a grid of bar plots, where each plot
    represents the distribution of a categorical feature, grouped by the target
    variable. It shows percentages and uses a consistent y-axis scale across all
    subplots.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be plotted.
        features (List[str]): A list of column names representing the categorical
        features to be plotted.
        target (str): The name of the target variable column used for grouping.
        save_path (Optional[str]): The file path to save the plot image. If None,
        the plot is not saved.

    Returns:
        None. The function displays the plot and optionally saves it to a file.
    """
    num_features = len(features)
    if num_features == 1:
        rows, cols = 1, 1
    elif num_features == 2:
        rows, cols = 1, 2
    elif num_features == 3:
        rows, cols = 2, 2
    else:
        rows, cols = (num_features + 1) // 2, 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )

    axis_font = {"family": "Styrene A", "color": "#191919"}
    colors = {str(i): color for i, color in enumerate(PRIMARY_COLORS)}

    for i, feature in enumerate(features):
        row, col = (i // cols) + 1, (i % cols) + 1
        data = (
            df.groupby([feature, target], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        data_percentages = data.div(data.sum(axis=1), axis=0) * 100

        for category in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data_percentages[category],
                    name=f"{target} = {category}",
                    marker_color=colors[str(category)],
                    text=[f"{v:.1f}%" for v in data_percentages[category]],
                    textposition="inside",
                    width=0.35,
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            title_text=feature,
            row=row,
            col=col,
            title_standoff=25,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            showticklabels=True,
        )

        fig.update_yaxes(
            title_text="Percentage" if col == 1 else None,
            row=row,
            col=col,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            range=[0, 100],
        )

    fig.update_layout(
        title_text=f"Distribution of {', '.join(features)} by {target}",
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=True,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=400 * rows,
        width=600 * cols,
        margin={"l": 50, "r": 150, "t": 100, "b": 50},
        font={**axis_font, "size": 12},
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="Black",
            borderwidth=1,
            font={**axis_font, "size": 12},
        ),
    )

    if save_path:
        fig.write_image(save_path)


def plot_numeric_distributions(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    nbins: int = 40,
    save_path: Optional[str] = None,
) -> None:
    """Plots numerical distribution for specified features in the DataFrame.

    Shows distributions for overall (sum of target 0 and 1), target = 0, and target = 1.
    Bars are plotted with overlap, using edge highlighting for distinction.
    Legend is placed on the right side with colors unaffected by opacity.

    Args:
        df: DataFrame containing the features and target variable.
        features: List of feature names to plot distributions for.
        target: Name of the binary target variable column.
        nbins: Number of bins for each histogram. Defaults to 40.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    rows, cols = 1, len(features)
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05)
    axis_font = {"family": "Styrene A", "color": "#191919"}
    plot_colors = PLOT_COLORS
    categories = ["Overall", "Target = 1", "Target = 0"]

    for i, feature in enumerate(features):
        hist_data = []
        bin_edges = None
        for category in [0, 1]:
            x = df[df[target] == category][feature]
            hist, edges = np.histogram(
                x, bins=nbins, range=(df[feature].min(), df[feature].max())
            )
            hist_data.append(hist)
            if bin_edges is None:
                bin_edges = edges

        overall_hist = hist_data[0] + hist_data[1]
        hist_data = [overall_hist] + hist_data[::-1]

        bin_width = (bin_edges[-1] - bin_edges[0]) / nbins

        for data, color, name in zip(hist_data, plot_colors, categories):
            fig.add_trace(
                go.Bar(
                    x=bin_edges[:-1],
                    y=data,
                    name=name,
                    marker_color=color,
                    opacity=0.1,
                    showlegend=False,
                    width=bin_width,
                ),
                row=1,
                col=i + 1,
            )

            fig.add_trace(
                go.Scatter(
                    x=bin_edges[:-1],
                    y=data,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=(i == 0),
                    name=name,
                    hoverinfo="skip",
                ),
                row=1,
                col=i + 1,
            )

        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )

        if i == 0:
            fig.update_yaxes(
                title_text="Count",
                row=1,
                col=1,
                title_font={**axis_font, "size": 14},
                tickfont={**axis_font, "size": 12},
            )

    fig.update_layout(
        title_text=f"Distribution of {', '.join(features)} by {target}",
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=600,
        width=400 * len(features) + 150,
        margin={
            "l": 50,
            "r": 150,
            "t": 80,
            "b": 80,
        },
        font={**axis_font, "size": 12},
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
    )

    fig.show()
    if save_path:
        fig.write_image(save_path)


def plot_single_bar_chart(
    df: pd.DataFrame,
    feature: str,
    save_path: Optional[str] = None,
) -> None:
    """Plots a percentage bar chart for a specified categorical feature in the DataFrame.

    Args:
        df: DataFrame containing the feature to plot.
        feature: Name of the categorical feature to plot.
        save_path: Optional path to save the plot image.

    Returns:
        None. Displays the plot and optionally saves it to a file.
    """
    title = f"Distribution of target variable {feature}"

    value_counts = df[feature].value_counts(normalize=True).reset_index()
    value_counts.columns = [feature, "percentage"]
    value_counts["percentage"] *= 100

    fig = go.Figure()

    for i, (value, percentage) in enumerate(
        zip(value_counts[feature], value_counts["percentage"])
    ):
        fig.add_trace(
            go.Bar(
                x=[value],
                y=[percentage],
                name=f"{feature} = {value}",
                marker_color=PRIMARY_COLORS[i % 2],
                text=[f"{percentage:.1f}%"],
                textposition="auto",
            )
        )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    fig.update_xaxes(
        title_text=feature,
        title_standoff=25,
        title_font={**axis_font, "size": 14},
        tickfont={**axis_font, "size": 12},
        showticklabels=True,
    )

    fig.update_yaxes(
        title_text="Percentage",
        title_font={**axis_font, "size": 14},
        tickfont={**axis_font, "size": 12},
        range=[0, 100],
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=True,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=1200,
        margin={"l": 50, "r": 150, "t": 80, "b": 50},
        font={**axis_font, "size": 12},
        barmode="group",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="Black",
            borderwidth=1,
            font={**axis_font, "size": 12},
        ),
    )
    if save_path:
        fig.write_image(save_path)
