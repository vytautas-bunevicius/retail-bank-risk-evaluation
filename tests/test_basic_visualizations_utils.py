"""
Test module for retail_bank_risk basic visualization utilities.

This module contains unit tests for various visualization functions
used in the retail bank risk assessment project. The functions are
designed to generate plots for analyzing both numerical and categorical
data distributions, correlations, and feature importances.

The module tests the following main functionalities:
1. Combined histograms for numerical features
2. Combined bar charts for categorical features
3. Boxplots for numerical features
4. Correlation matrix visualization
5. Feature importances for models
6. Distribution comparison between datasets
7. Categorical feature analysis by target variable
8. Numeric feature distributions by target variable
9. Single bar chart generation

Mocking is used to test the plot generation and ensure that no image
files are written during the tests. Each test function corresponds to
a specific utility function from the retail_bank_risk.basic_visualizations_utils
module, ensuring these functions behave as expected with sample data.
"""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from retail_bank_risk.basic_visualizations_utils import (
    plot_categorical_features_by_target,
    plot_combined_bar_charts,
    plot_combined_boxplots,
    plot_combined_histograms,
    plot_correlation_matrix,
    plot_distribution_comparison,
    plot_feature_importances,
    plot_numeric_distributions,
    plot_single_bar_chart,
)


class TestBasicVisualizationsUtils(unittest.TestCase):
    """
    Test suite for basic_visualizations_utils module functions.
    """

    def setUp(self) -> None:
        """
        Set up a sample DataFrame and feature lists for testing.
        """
        np.random.seed(42)
        self.df: pd.DataFrame = pd.DataFrame(
            {
                "numeric1": np.random.normal(0, 1, 1000),
                "numeric2": np.random.normal(5, 2, 1000),
                "categorical1": np.random.choice(["A", "B", "C"], 1000),
                "categorical2": np.random.choice(["X", "Y", "Z"], 1000),
                "target": np.random.choice([0, 1], 1000),
            }
        )
        self.numeric_features: list[str] = ["numeric1", "numeric2"]
        self.categorical_features: list[str] = ["categorical1", "categorical2"]

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    @patch("retail_bank_risk.basic_visualizations_utils.go.Histogram")
    def test_plot_combined_histograms(
        self, mock_histogram: MagicMock, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_combined_histograms function to ensure it creates the correct
        number of histogram traces and does not call write_image.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_histograms(self.df, self.numeric_features)

        self.assertEqual(mock_histogram.call_count, len(self.numeric_features))
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    @patch("retail_bank_risk.basic_visualizations_utils.go.Bar")
    def test_plot_combined_bar_charts(
        self, mock_bar: MagicMock, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_combined_bar_charts function to ensure it creates the correct
        number of bar traces, shows the figure, and does not call write_image.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_bar_charts(self.df, self.categorical_features)

        self.assertEqual(mock_bar.call_count, len(self.categorical_features))
        mock_fig.show.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    @patch("retail_bank_risk.basic_visualizations_utils.go.Box")
    def test_plot_combined_boxplots(
        self, mock_box: MagicMock, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_combined_boxplots function to ensure it creates the correct
        number of boxplot traces and does not call write_image.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_combined_boxplots(self.df, self.numeric_features)

        self.assertEqual(mock_box.call_count, len(self.numeric_features))
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.px.imshow")
    def test_plot_correlation_matrix(self, mock_imshow: MagicMock) -> None:
        """
        Test the plot_correlation_matrix function to ensure imshow is called once
        and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_imshow.return_value = mock_fig

        plot_correlation_matrix(self.df, self.numeric_features)

        mock_imshow.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.go.Figure")
    def test_plot_feature_importances(self, mock_figure: MagicMock) -> None:
        """
        Test the plot_feature_importances function to ensure a figure is created,
        shown, and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_figure.return_value = mock_fig

        feature_importances: Dict[str, Dict[str, float]] = {
            "Model1": {"numeric1": 0.5, "numeric2": 0.3},
            "Model2": {"numeric1": 0.4, "numeric2": 0.6},
        }

        plot_feature_importances(feature_importances)

        mock_figure.assert_called_once()
        mock_fig.show.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    def test_plot_distribution_comparison(
        self, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_distribution_comparison function to ensure subplots are created,
        the figure is shown, and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        df_before: pd.DataFrame = self.df.copy()
        df_after: pd.DataFrame = self.df.copy()
        df_after["numeric1"] = df_after["numeric1"] + 1

        plot_distribution_comparison(df_before, df_after, self.numeric_features)

        mock_make_subplots.assert_called_once()
        mock_fig.show.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    def test_plot_categorical_features_by_target(
        self, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_categorical_features_by_target function to ensure subplots
        are created and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_categorical_features_by_target(
            self.df, self.categorical_features, "target"
        )

        mock_make_subplots.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.make_subplots")
    def test_plot_numeric_distributions(
        self, mock_make_subplots: MagicMock
    ) -> None:
        """
        Test the plot_numeric_distributions function to ensure subplots are created,
        the figure is shown, and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_make_subplots.return_value = mock_fig

        plot_numeric_distributions(self.df, self.numeric_features, "target")

        mock_make_subplots.assert_called_once()
        mock_fig.show.assert_called_once()
        mock_fig.write_image.assert_not_called()

    @patch("retail_bank_risk.basic_visualizations_utils.go.Figure")
    def test_plot_single_bar_chart(self, mock_figure: MagicMock) -> None:
        """
        Test the plot_single_bar_chart function to ensure a figure is created
        and write_image is not called.
        """
        mock_fig: MagicMock = MagicMock()
        mock_figure.return_value = mock_fig

        plot_single_bar_chart(self.df, "categorical1")

        mock_figure.assert_called_once()
        mock_fig.write_image.assert_not_called()

    def test_input_validation(self) -> None:
        """
        Test input validation for various visualization functions to ensure they
        raise appropriate exceptions when given invalid inputs.
        """
        with self.assertRaises(KeyError):
            plot_combined_histograms(self.df, ["non_existent_column"])

        with self.assertRaises(ValueError):
            plot_combined_bar_charts(self.df, [])

        with self.assertRaises(AttributeError):
            plot_feature_importances("not a dict")


if __name__ == "__main__":
    unittest.main()
