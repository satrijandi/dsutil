"""
Tests for feature_engineering.py module
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the pipeline directory to Python path
pipeline_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pipeline_path)

# Mock lightgbm at module level to avoid import errors
with patch.dict("sys.modules", {"lightgbm": MagicMock(), "scipy": MagicMock(), "scipy.stats": MagicMock()}):
    from utils.feature_engineering import detect_shifting_features, feature_selection_lgbm


class TestFeatureSelectionLgbm:
    """Test feature selection using LightGBM"""

    def test_feature_selection_basic(self):
        """Test basic feature selection functionality"""
        # Create sample data
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10], "feature3": [1, 1, 1, 1, 1]}  # constant feature
        )
        y = pd.Series([0, 1, 0, 1, 0])

        with patch("lightgbm.Dataset") as mock_dataset, patch("lightgbm.train") as mock_train:

            # Mock the model and its methods
            mock_model = MagicMock()
            mock_model.feature_importance.return_value = [10, 5, 1]  # importance scores
            mock_train.return_value = mock_model

            # Test feature selection
            selected_features = feature_selection_lgbm(X, y, n_features=2)

            # Verify results
            assert isinstance(selected_features, list)
            assert len(selected_features) == 2

            # Verify LightGBM was called
            mock_dataset.assert_called_once()
            mock_train.assert_called_once()

    def test_feature_selection_empty_data(self):
        """Test feature selection with empty data"""
        X = pd.DataFrame()
        y = pd.Series([])

        with patch("lightgbm.Dataset"), patch("lightgbm.train") as mock_train:
            mock_train.side_effect = ValueError("Empty dataset")

            with pytest.raises(ValueError):
                feature_selection_lgbm(X, y, n_features=5)


class TestDetectShiftingFeatures:
    """Test drift detection functionality"""

    def test_detect_shifting_features_basic(self):
        """Test basic drift detection"""
        # Create test data with temporal component
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        train_df = pd.DataFrame(
            {
                "date_col": dates,
                "feature1": range(60),  # trending feature (likely to show drift)
                "feature2": [1] * 60,  # stable feature
                "target": [0, 1] * 30,
                "id_col": range(60),
            }
        )

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        # Mock the parallel processing to avoid complexity in tests
        with patch("utils.feature_engineering.ProcessPoolExecutor") as mock_executor:
            mock_context = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_context
            # Mock results indicating feature1 is shifting, feature2 is stable
            mock_context.map.return_value = [
                ("feature1", 0.4, True),  # high KS stat, is shifting
                ("feature2", 0.1, False),  # low KS stat, not shifting
            ]

            shifting_features = detect_shifting_features(train_df, config, threshold=0.3)

            assert isinstance(shifting_features, list)
            # Based on our mock, should detect feature1 as shifting
            assert "feature1" in shifting_features

    def test_detect_shifting_features_insufficient_months(self):
        """Test drift detection with insufficient data"""
        # Create data with only one month
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        train_df = pd.DataFrame({"date_col": dates, "feature1": range(5), "target": [0, 1, 0, 1, 0], "id_col": range(5)})

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        shifting_features = detect_shifting_features(train_df, config)

        # Should return empty list due to insufficient months
        assert shifting_features == []

    def test_detect_shifting_features_no_data(self):
        """Test drift detection with empty dataframe"""
        train_df = pd.DataFrame()
        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        shifting_features = detect_shifting_features(train_df, config)
        assert shifting_features == []
