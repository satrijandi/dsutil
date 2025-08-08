"""
Comprehensive unit tests for feature_engineering.py module with full mocking
"""

import os
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)


class TestFeatureSelectionLgbmComprehensive:
    """Test feature selection using LightGBM with comprehensive coverage"""

    @patch("utils.feature_engineering.lgb")
    def test_feature_selection_lgbm_basic(self, mock_lgb):
        """Test basic feature selection functionality with full mocking"""
        from utils.feature_engineering import feature_selection_lgbm

        # Mock LightGBM Dataset
        mock_dataset = MagicMock()
        mock_lgb.Dataset.return_value = mock_dataset

        # Mock LightGBM model
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([10, 5, 1, 8, 3])
        mock_lgb.train.return_value = mock_model

        # Mock early stopping and log evaluation callbacks
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        # Create sample data
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "feature3": [1, 1, 1, 1, 1],
                "feature4": [3, 6, 9, 12, 15],
                "feature5": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        result = feature_selection_lgbm(X, y, n_features=3)

        # Verify LightGBM components were called
        mock_lgb.Dataset.assert_called_once_with(X, label=y)
        mock_lgb.train.assert_called_once()

        # Verify training parameters
        train_call = mock_lgb.train.call_args
        params = train_call[0][0]
        assert params["objective"] == "binary"
        assert params["metric"] == "auc"
        assert params["boosting_type"] == "gbdt"

        # Verify callbacks
        callbacks = train_call[1]["callbacks"]
        assert len(callbacks) == 2

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 3
        # Should select top 3 features based on importance: feature1 (10), feature4 (8), feature2 (5)
        assert "feature1" in result
        assert "feature4" in result
        assert "feature2" in result

    @patch("utils.feature_engineering.lgb")
    def test_feature_selection_lgbm_custom_parameters(self, mock_lgb):
        """Test feature selection with custom parameters"""
        from utils.feature_engineering import feature_selection_lgbm

        mock_dataset = MagicMock()
        mock_lgb.Dataset.return_value = mock_dataset

        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([1, 2, 3])
        mock_lgb.train.return_value = mock_model

        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        y = pd.Series([0, 1])

        result = feature_selection_lgbm(X, y, n_features=2)

        # Verify parameters are correctly set
        train_call = mock_lgb.train.call_args
        params = train_call[0][0]

        expected_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        for key, value in expected_params.items():
            assert params[key] == value

        assert len(result) == 2

    @patch("utils.feature_engineering.lgb")
    def test_feature_selection_lgbm_single_feature(self, mock_lgb):
        """Test feature selection requesting single feature"""
        from utils.feature_engineering import feature_selection_lgbm

        mock_lgb.Dataset.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([100])
        mock_lgb.train.return_value = mock_model
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        X = pd.DataFrame({"single_feature": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        result = feature_selection_lgbm(X, y, n_features=1)

        assert len(result) == 1
        assert result[0] == "single_feature"

    @patch("utils.feature_engineering.lgb")
    def test_feature_selection_lgbm_more_features_than_available(self, mock_lgb):
        """Test requesting more features than available"""
        from utils.feature_engineering import feature_selection_lgbm

        mock_lgb.Dataset.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([10, 5])
        mock_lgb.train.return_value = mock_model
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        X = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
        y = pd.Series([0, 1])

        result = feature_selection_lgbm(X, y, n_features=10)

        # Should return all available features
        assert len(result) == 2
        assert set(result) == {"feat1", "feat2"}

    @patch("utils.feature_engineering.lgb")
    def test_feature_selection_lgbm_training_failure(self, mock_lgb):
        """Test handling of training failures"""
        from utils.feature_engineering import feature_selection_lgbm

        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.train.side_effect = Exception("Training failed")
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        X = pd.DataFrame({"feat1": [1, 2]})
        y = pd.Series([0, 1])

        with pytest.raises(Exception, match="Training failed"):
            feature_selection_lgbm(X, y, n_features=1)


class TestDetectShiftingFeaturesComprehensive:
    """Test drift detection functionality with comprehensive coverage"""

    @patch("utils.feature_engineering.ProcessPoolExecutor")
    def test_detect_shifting_features_basic(self, mock_executor):
        """Test basic drift detection functionality"""
        from utils.feature_engineering import detect_shifting_features

        # Mock ProcessPoolExecutor
        mock_context = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_context

        # Mock results - feature1 is shifting, feature2 is stable
        mock_results = [
            ("feature1", 0.4, True),  # KS stat > threshold, shifting
            ("feature2", 0.1, False),  # KS stat < threshold, stable
        ]
        mock_context.map.return_value = mock_results

        # Create test data with multiple months
        dates = pd.date_range("2024-01-01", periods=120, freq="D")  # 4 months
        train_df = pd.DataFrame(
            {
                "date_col": dates,
                "feature1": range(120),  # Trending feature
                "feature2": [1] * 120,  # Stable feature
                "target": [0, 1] * 60,
                "id_col": range(120),
            }
        )

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        result = detect_shifting_features(train_df, config, threshold=0.3)

        # Verify ProcessPoolExecutor was used
        mock_executor.assert_called()
        mock_context.map.assert_called()

        # Verify result contains shifting features
        assert isinstance(result, list)
        assert "feature1" in result
        assert "feature2" not in result

    @patch("utils.feature_engineering.ProcessPoolExecutor")
    def test_detect_shifting_features_no_shifting(self, mock_executor):
        """Test when no features are shifting"""
        from utils.feature_engineering import detect_shifting_features

        mock_context = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_context

        # All features stable
        mock_results = [
            ("feature1", 0.1, False),
            ("feature2", 0.2, False),
        ]
        mock_context.map.return_value = mock_results

        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        train_df = pd.DataFrame(
            {"date_col": dates, "feature1": [1] * 90, "feature2": [2] * 90, "target": [0, 1] * 45, "id_col": range(90)}
        )

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        result = detect_shifting_features(train_df, config, threshold=0.3)

        assert result == []

    def test_detect_shifting_features_empty_dataframe(self):
        """Test drift detection with empty dataframe"""
        from utils.feature_engineering import detect_shifting_features

        train_df = pd.DataFrame()
        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        result = detect_shifting_features(train_df, config)
        assert result == []

    def test_detect_shifting_features_missing_date_column(self):
        """Test drift detection with missing date column"""
        from utils.feature_engineering import detect_shifting_features

        train_df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "missing_date_col", "TARGET_COL": "target", "ID_COLUMNS": []}

        result = detect_shifting_features(train_df, config)
        assert result == []

    def test_detect_shifting_features_insufficient_months(self):
        """Test drift detection with insufficient months"""
        from utils.feature_engineering import detect_shifting_features

        # Only one month of data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        train_df = pd.DataFrame({"date_col": dates, "feature1": range(10), "target": [0, 1] * 5, "id_col": range(10)})

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        result = detect_shifting_features(train_df, config)
        assert result == []

    @patch("utils.feature_engineering.ProcessPoolExecutor")
    def test_detect_shifting_features_custom_threshold(self, mock_executor):
        """Test drift detection with custom threshold"""
        from utils.feature_engineering import detect_shifting_features

        mock_context = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_context

        # Feature with KS stat between default and custom threshold
        mock_results = [("feature1", 0.25, False)]  # Would be shifting with threshold 0.2
        mock_context.map.return_value = mock_results

        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        train_df = pd.DataFrame({"date_col": dates, "feature1": range(60), "target": [0, 1] * 30, "id_col": range(60)})

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        # With higher threshold, feature should not be detected
        result = detect_shifting_features(train_df, config, threshold=0.3)
        assert result == []

    @patch("utils.feature_engineering.ProcessPoolExecutor")
    def test_detect_shifting_features_multiple_months(self, mock_executor):
        """Test drift detection across multiple months"""
        from utils.feature_engineering import detect_shifting_features

        mock_context = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_context

        # Simulate results from multiple month comparisons
        mock_results_calls = [
            [("feature1", 0.2, False)],  # Month 2 vs reference
            [("feature1", 0.4, True)],  # Month 3 vs reference
            [("feature1", 0.5, True)],  # Month 4 vs reference
        ]
        mock_context.map.side_effect = mock_results_calls

        # Create 4 months of data
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        train_df = pd.DataFrame(
            {
                "date_col": dates,
                "feature1": list(range(30)) * 4,  # Cycling pattern
                "target": [0, 1] * 60,
                "id_col": range(120),
            }
        )

        config = {"TRAIN_START_DATE": "2024-01-01", "DATE_COL": "date_col", "TARGET_COL": "target", "ID_COLUMNS": ["id_col"]}

        result = detect_shifting_features(train_df, config, threshold=0.3)

        # Should detect shifting feature (found in month 3 and 4)
        assert "feature1" in result

        # Should have called executor.map multiple times (once per month comparison)
        assert mock_context.map.call_count >= 2

    def test_calculate_feature_drift_helper(self):
        """Test the helper function for drift calculation"""
        from utils.feature_engineering import _calculate_feature_drift

        # Create test data with clear distribution difference
        reference_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5] * 20})
        current_data = pd.DataFrame({"feature1": [6, 7, 8, 9, 10] * 20})

        args = ("feature1", reference_data, current_data, 0.3)

        feature, ks_stat, is_shifting = _calculate_feature_drift(args)

        assert feature == "feature1"
        assert isinstance(ks_stat, float)
        assert ks_stat > 0
        assert isinstance(is_shifting, bool)

    def test_calculate_feature_drift_helper_empty_data(self):
        """Test drift calculation helper with empty data"""
        from utils.feature_engineering import _calculate_feature_drift

        reference_data = pd.DataFrame({"feature1": []})
        current_data = pd.DataFrame({"feature1": [1, 2, 3]})

        args = ("feature1", reference_data, current_data, 0.3)

        feature, ks_stat, is_shifting = _calculate_feature_drift(args)

        assert feature == "feature1"
        assert ks_stat == 0.0
        assert is_shifting is False

    def test_calculate_feature_drift_helper_with_nan(self):
        """Test drift calculation helper with NaN values"""
        from utils.feature_engineering import _calculate_feature_drift

        reference_data = pd.DataFrame({"feature1": [1, 2, np.nan, 4, 5]})
        current_data = pd.DataFrame({"feature1": [6, np.nan, 8, 9, 10]})

        args = ("feature1", reference_data, current_data, 0.3)

        feature, ks_stat, is_shifting = _calculate_feature_drift(args)

        assert feature == "feature1"
        assert isinstance(ks_stat, float)
        assert isinstance(is_shifting, bool)

    def test_calculate_feature_drift_helper_exception(self):
        """Test drift calculation helper with exception"""
        from utils.feature_engineering import _calculate_feature_drift

        # Create data that might cause issues
        reference_data = pd.DataFrame({"feature1": ["a", "b", "c"]})  # String data
        current_data = pd.DataFrame({"feature1": ["x", "y", "z"]})

        args = ("feature1", reference_data, current_data, 0.3)

        feature, ks_stat, is_shifting = _calculate_feature_drift(args)

        # Should handle exception gracefully
        assert feature == "feature1"
        assert ks_stat == 0.0
        assert is_shifting is False
