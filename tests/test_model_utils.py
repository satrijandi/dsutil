"""
Comprehensive unit tests for model_utils.py module
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


class TestTrainH2OAutoML:
    """Test H2O AutoML training functionality"""

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.H2OAutoML")
    @patch("utils.model_utils.os.cpu_count")
    def test_train_h2o_automl_basic(self, mock_cpu_count, mock_automl_class, mock_h2o):
        """Test basic H2O AutoML training"""
        from utils.model_utils import train_h2o_automl

        mock_cpu_count.return_value = 8

        # Mock H2O frames
        mock_train_frame = MagicMock()
        mock_test_frame = MagicMock()
        mock_h2o.H2OFrame.side_effect = [mock_train_frame, mock_test_frame]

        # Mock AutoML
        mock_aml = MagicMock()
        mock_automl_class.return_value = mock_aml

        # Test data
        train_df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5], "target": [0, 1, 0, 1, 0]}
        )

        test_df = pd.DataFrame({"feature1": [6, 7, 8], "feature2": [0.6, 0.7, 0.8], "target": [1, 0, 1]})

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}

        selected_features = ["feature1", "feature2", "target"]

        result = train_h2o_automl(train_df, test_df, config, selected_features)

        # Verify H2O initialization
        mock_h2o.init.assert_called_once()

        # Verify AutoML creation and training
        mock_automl_class.assert_called_once()
        mock_aml.train.assert_called_once()

        assert result == mock_aml

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.H2OAutoML")
    @patch("utils.model_utils.os.cpu_count")
    def test_train_h2o_automl_with_mlflow(self, mock_cpu_count, mock_automl_class, mock_h2o):
        """Test H2O AutoML training with MLflow logging"""
        from utils.model_utils import train_h2o_automl

        mock_cpu_count.return_value = 4

        # Mock MLflow manager
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.create_model_signature.return_value = "signature"
        mock_mlflow_manager.log_model.return_value = MagicMock()

        # Mock H2O frames
        mock_train_frame = MagicMock()
        mock_test_frame = MagicMock()
        mock_h2o.H2OFrame.side_effect = [mock_train_frame, mock_test_frame]

        # Mock AutoML
        mock_aml = MagicMock()
        mock_automl_class.return_value = mock_aml

        train_df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        test_df = pd.DataFrame({"feature1": [3, 4], "target": [1, 0]})
        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target", "PROJECT_NAME": "test_project"}
        selected_features = ["feature1", "target"]

        result = train_h2o_automl(train_df, test_df, config, selected_features, mock_mlflow_manager)

        # Verify MLflow operations
        mock_mlflow_manager.log_parameters.assert_called_once()
        mock_mlflow_manager.create_model_signature.assert_called_once()
        mock_mlflow_manager.log_model.assert_called_once()

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.H2OAutoML")
    def test_train_h2o_automl_data_size_scaling(self, mock_automl_class, mock_h2o):
        """Test that AutoML parameters scale with data size"""
        from utils.model_utils import train_h2o_automl

        # Mock H2O frames
        mock_h2o.H2OFrame.return_value = MagicMock()
        mock_aml = MagicMock()
        mock_automl_class.return_value = mock_aml

        # Test small dataset
        small_train_df = pd.DataFrame({"feature1": range(5000), "target": [0, 1] * 2500})  # Small dataset < 10000
        test_df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = ["feature1", "target"]

        train_h2o_automl(small_train_df, test_df, config, selected_features)

        # Verify small dataset gets fewer models and less time
        call_args = mock_automl_class.call_args[1]
        assert call_args["max_models"] == 8
        assert call_args["max_runtime_secs"] == 600

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.H2OAutoML")
    def test_train_h2o_automl_feature_scaling(self, mock_automl_class, mock_h2o):
        """Test that runtime scales with number of features"""
        from utils.model_utils import train_h2o_automl

        mock_h2o.H2OFrame.return_value = MagicMock()
        mock_aml = MagicMock()
        mock_automl_class.return_value = mock_aml

        # Create dataset with many features
        feature_data = {f"feature{i}": [1, 2, 3] for i in range(150)}
        feature_data["target"] = [0, 1, 0]
        train_df = pd.DataFrame(feature_data)
        test_df = pd.DataFrame({f"feature{i}": [4, 5] for i in range(150)})
        test_df["target"] = [1, 0]

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = list(feature_data.keys())

        train_h2o_automl(train_df, test_df, config, selected_features)

        # Verify high-dimensional data gets more runtime
        call_args = mock_automl_class.call_args[1]
        assert call_args["max_runtime_secs"] > 600  # Should be increased for many features


class TestEvaluateModel:
    """Test model evaluation functionality"""

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.roc_auc_score")
    def test_evaluate_model_basic(self, mock_roc_auc, mock_h2o):
        """Test basic model evaluation"""
        from utils.model_utils import evaluate_model

        # Mock ROC AUC scores
        mock_roc_auc.side_effect = [0.85, 0.82, 0.88]  # Different AUC for each month

        # Mock H2O model
        mock_aml = MagicMock()
        mock_leader = MagicMock()
        mock_aml.leader = mock_leader
        mock_leader.auc.return_value = 0.85
        mock_leader.varimp.return_value = pd.DataFrame(
            {"variable": ["feature1", "feature2"], "relative_importance": [1.0, 0.5]}
        )

        # Mock H2O frame and predictions
        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        mock_predictions = MagicMock()
        mock_predictions.as_data_frame.return_value = pd.DataFrame({"p1": [0.7, 0.8, 0.6]})
        mock_leader.predict.return_value = mock_predictions

        # Test data
        train_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [0, 1, 0],
                "date_col": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]),
            }
        )

        test_df = pd.DataFrame(
            {
                "feature1": [4, 5, 6],
                "target": [1, 0, 1],
                "date_col": pd.to_datetime(["2024-04-15", "2024-05-15", "2024-06-15"]),
            }
        )

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}

        selected_features = ["feature1", "target"]

        result = evaluate_model(mock_aml, train_df, test_df, config, selected_features)

        # Verify results structure
        assert "monthly_performance" in result
        assert "feature_importance" in result
        assert "model_summary" in result

        # Verify monthly performance is a DataFrame
        assert isinstance(result["monthly_performance"], pd.DataFrame)
        assert len(result["monthly_performance"]) > 0

        # Verify feature importance
        assert isinstance(result["feature_importance"], pd.DataFrame)

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.roc_auc_score")
    def test_evaluate_model_with_mlflow(self, mock_roc_auc, mock_h2o):
        """Test model evaluation with MLflow logging"""
        from utils.model_utils import evaluate_model

        mock_roc_auc.return_value = 0.85

        # Mock MLflow manager
        mock_mlflow_manager = MagicMock()

        # Mock H2O model
        mock_aml = MagicMock()
        mock_leader = MagicMock()
        mock_aml.leader = mock_leader
        mock_leader.auc.return_value = 0.85
        mock_leader.varimp.return_value = pd.DataFrame({"variable": ["feature1"], "relative_importance": [1.0]})
        mock_aml.leaderboard.as_data_frame.return_value = pd.DataFrame({"model_id": ["model1"]})

        # Mock H2O operations
        mock_h2o.H2OFrame.return_value = MagicMock()
        mock_predictions = MagicMock()
        mock_predictions.as_data_frame.return_value = pd.DataFrame({"p1": [0.7]})
        mock_leader.predict.return_value = mock_predictions

        train_df = pd.DataFrame({"feature1": [1], "target": [0], "date_col": pd.to_datetime(["2024-01-15"])})
        test_df = pd.DataFrame({"feature1": [2], "target": [1], "date_col": pd.to_datetime(["2024-02-15"])})

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = ["feature1", "target"]

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            result = evaluate_model(mock_aml, train_df, test_df, config, selected_features, mock_mlflow_manager)

        # Verify MLflow operations
        mock_mlflow_manager.log_metrics.assert_called_once()
        mock_mlflow_manager.log_artifact.assert_called()

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.roc_auc_score")
    def test_evaluate_model_feature_importance_error(self, mock_roc_auc, mock_h2o):
        """Test model evaluation when feature importance fails"""
        from utils.model_utils import evaluate_model

        mock_roc_auc.return_value = 0.85

        # Mock H2O model with feature importance error
        mock_aml = MagicMock()
        mock_leader = MagicMock()
        mock_aml.leader = mock_leader
        mock_leader.auc.return_value = 0.85
        mock_leader.varimp.side_effect = Exception("Feature importance error")

        # Mock H2O operations
        mock_h2o.H2OFrame.return_value = MagicMock()
        mock_predictions = MagicMock()
        mock_predictions.as_data_frame.return_value = pd.DataFrame({"p1": [0.7]})
        mock_leader.predict.return_value = mock_predictions

        train_df = pd.DataFrame({"feature1": [1], "target": [0], "date_col": pd.to_datetime(["2024-01-15"])})
        test_df = pd.DataFrame({"feature1": [2], "target": [1], "date_col": pd.to_datetime(["2024-02-15"])})

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = ["feature1", "target"]

        result = evaluate_model(mock_aml, train_df, test_df, config, selected_features)

        # Should create fallback feature importance
        assert "feature_importance" in result
        feature_importance = result["feature_importance"]
        assert isinstance(feature_importance, pd.DataFrame)
        assert "variable" in feature_importance.columns
        assert "relative_importance" in feature_importance.columns

    @patch("utils.model_utils.h2o")
    @patch("utils.model_utils.roc_auc_score")
    def test_evaluate_model_cross_validation(self, mock_roc_auc, mock_h2o):
        """Test model evaluation with cross-validation metrics"""
        from utils.model_utils import evaluate_model

        mock_roc_auc.return_value = 0.85

        # Mock H2O model with cross-validation
        mock_aml = MagicMock()
        mock_leader = MagicMock()
        mock_aml.leader = mock_leader
        mock_leader.auc.return_value = 0.85
        mock_leader.varimp.return_value = pd.DataFrame({"variable": ["feature1"], "relative_importance": [1.0]})

        # Mock cross-validation summary
        cv_summary_df = pd.DataFrame({"metric": ["auc", "accuracy"], "mean": [0.82, 0.78], "std": [0.03, 0.05]})
        mock_cv_summary = MagicMock()
        mock_cv_summary.as_data_frame.return_value = cv_summary_df
        mock_leader.cross_validation_metrics_summary.return_value = mock_cv_summary

        # Mock confusion matrix
        cm_df = pd.DataFrame({"Actual": ["0", "1"], "0": [50, 10], "1": [5, 35]})
        mock_cm = MagicMock()
        mock_cm.as_data_frame.return_value = cm_df
        mock_leader.confusion_matrix.return_value = mock_cm

        mock_aml.leaderboard.as_data_frame.return_value = pd.DataFrame({"model_id": ["model1"]})

        # Mock H2O operations
        mock_h2o.H2OFrame.return_value = MagicMock()
        mock_predictions = MagicMock()
        mock_predictions.as_data_frame.return_value = pd.DataFrame({"p1": [0.7]})
        mock_leader.predict.return_value = mock_predictions

        train_df = pd.DataFrame({"feature1": [1], "target": [0], "date_col": pd.to_datetime(["2024-01-15"])})
        test_df = pd.DataFrame({"feature1": [2], "target": [1], "date_col": pd.to_datetime(["2024-02-15"])})

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = ["feature1", "target"]

        result = evaluate_model(mock_aml, train_df, test_df, config, selected_features)

        # Verify cross-validation metrics are included
        model_summary = result["model_summary"]
        cv_metrics = model_summary["performance_metrics"]["cross_validation_metrics"]

        assert "auc" in cv_metrics
        assert cv_metrics["auc"]["mean"] == 0.82
        assert cv_metrics["auc"]["std"] == 0.03

        # Verify confusion matrix is included
        assert "confusion_matrix" in model_summary

    @patch("utils.model_utils.h2o")
    def test_evaluate_model_empty_data(self, mock_h2o):
        """Test model evaluation with empty data"""
        from utils.model_utils import evaluate_model

        mock_aml = MagicMock()
        mock_leader = MagicMock()
        mock_aml.leader = mock_leader
        mock_leader.auc.return_value = 0.85
        mock_leader.varimp.return_value = pd.DataFrame({"variable": [], "relative_importance": []})

        # Empty dataframes
        train_df = pd.DataFrame({"date_col": pd.to_datetime([])})
        test_df = pd.DataFrame({"date_col": pd.to_datetime([])})

        config = {"ID_COLUMNS": [], "DATE_COL": "date_col", "TARGET_COL": "target"}
        selected_features = ["target"]

        result = evaluate_model(mock_aml, train_df, test_df, config, selected_features)

        # Should handle empty data gracefully
        assert "monthly_performance" in result
        assert isinstance(result["monthly_performance"], pd.DataFrame)
