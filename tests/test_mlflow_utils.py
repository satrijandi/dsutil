"""
Comprehensive unit tests for mlflow_utils.py module
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)


class TestMLflowManager:
    """Test MLflowManager functionality"""

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_mlflow_manager_initialization(self, mock_client, mock_mlflow):
        """Test MLflowManager initialization"""
        from utils.mlflow_utils import MLflowManager

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        manager = MLflowManager(tracking_uri="test_uri", experiment_name="test_experiment")

        assert manager.tracking_uri == "test_uri"
        assert manager.experiment_name == "test_experiment"
        assert manager.experiment_id == "test_exp_id"

        mock_mlflow.set_tracking_uri.assert_called_once_with("test_uri")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_create_new_experiment(self, mock_client, mock_mlflow):
        """Test creating new experiment when it doesn't exist"""
        from utils.mlflow_utils import MLflowManager

        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp_id"

        manager = MLflowManager(experiment_name="new_experiment")

        mock_mlflow.create_experiment.assert_called_once_with("new_experiment")
        assert manager.experiment_id == "new_exp_id"

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_start_run(self, mock_client, mock_mlflow):
        """Test starting MLflow run"""
        from utils.mlflow_utils import MLflowManager

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        manager = MLflowManager()
        result = manager.start_run(run_name="test_run", tags={"key": "value"})

        mock_mlflow.start_run.assert_called_once_with(run_name="test_run", tags={"key": "value"})
        assert result == mock_run

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_parameters_small_set(self, mock_client, mock_mlflow):
        """Test logging small parameter set"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        params = {"param1": "value1", "param2": 2}

        manager.log_parameters(params)

        mock_mlflow.log_params.assert_called_once_with(params)

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_parameters_large_set(self, mock_client, mock_mlflow):
        """Test logging large parameter set (batching)"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        # Create large parameter set
        params = {f"param{i}": f"value{i}" for i in range(150)}

        manager.log_parameters(params)

        # Should be called multiple times due to batching
        assert mock_mlflow.log_params.call_count > 1

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_metrics_valid(self, mock_client, mock_mlflow):
        """Test logging valid metrics"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        metrics = {"accuracy": 0.85, "precision": 0.78, "recall": 0.82}

        manager.log_metrics(metrics, step=1)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=1)

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_metrics_invalid_filtered(self, mock_client, mock_mlflow):
        """Test logging metrics with invalid values filtered out"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        metrics = {
            "accuracy": 0.85,
            "invalid_nan": float("nan"),
            "invalid_inf": float("inf"),
            "invalid_neg_inf": float("-inf"),
            "valid_float": 0.78,
        }

        manager.log_metrics(metrics)

        expected_metrics = {"accuracy": 0.85, "valid_float": 0.78}
        mock_mlflow.log_metrics.assert_called_once_with(expected_metrics, step=None)

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_artifact(self, mock_client, mock_mlflow):
        """Test logging artifact"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        manager.log_artifact("test_file.txt", "artifacts")

        mock_mlflow.log_artifact.assert_called_once_with("test_file.txt", "artifacts")

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_log_model_with_input_example(self, mock_client, mock_mlflow):
        """Test logging H2O model with input example"""
        from utils.mlflow_utils import MLflowManager

        mock_model = MagicMock()
        mock_model.leader = MagicMock()

        mock_model_info = MagicMock()
        mock_mlflow.h2o.log_model.return_value = mock_model_info

        input_example = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5, 6, 7, 8], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        )

        manager = MLflowManager()
        result = manager.log_model(
            model=mock_model, model_name="test_model", input_example=input_example, registered_model_name="registered_model"
        )

        # Should limit input example to 5 rows
        call_args = mock_mlflow.h2o.log_model.call_args
        logged_input_example = call_args[1]["input_example"]
        assert len(logged_input_example) == 5
        assert result == mock_model_info

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_register_model_new(self, mock_client_class, mock_mlflow):
        """Test registering new model"""
        from utils.mlflow_utils import MLflowManager

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate model doesn't exist
        mock_client.get_registered_model.side_effect = Exception("Not found")

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.create_model_version.return_value = mock_version

        manager = MLflowManager()
        result = manager.register_model(model_uri="runs:/123/model", model_name="new_model", description="Test model")

        mock_client.create_registered_model.assert_called_once()
        mock_client.create_model_version.assert_called_once()
        assert result == mock_version

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_get_model_version_by_stage(self, mock_client_class, mock_mlflow):
        """Test getting model version by stage"""
        from utils.mlflow_utils import MLflowManager

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_version = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_version]

        manager = MLflowManager()
        result = manager.get_model_version("test_model", stage="Production")

        mock_client.get_latest_versions.assert_called_once_with("test_model", stages=["Production"])
        assert result == mock_version

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_load_model(self, mock_client, mock_mlflow):
        """Test loading model from MLflow"""
        from utils.mlflow_utils import MLflowManager

        mock_model = MagicMock()
        mock_mlflow.h2o.load_model.return_value = mock_model

        manager = MLflowManager()
        result = manager.load_model("models:/test_model/1")

        mock_mlflow.h2o.load_model.assert_called_once_with("models:/test_model/1")
        assert result == mock_model

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_transition_model_stage(self, mock_client_class, mock_mlflow):
        """Test transitioning model stage"""
        from utils.mlflow_utils import MLflowManager

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = MLflowManager()
        manager.transition_model_stage("test_model", "1", "Production")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model", version="1", stage="Production", archive_existing_versions=True
        )

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_get_best_run(self, mock_client_class, mock_mlflow):
        """Test getting best run by metric"""
        from utils.mlflow_utils import MLflowManager

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create mock runs with metrics
        mock_run1 = MagicMock()
        mock_run1.data.metrics = {"auc": 0.85}
        mock_run1.info.run_id = "run1"

        mock_run2 = MagicMock()
        mock_run2.data.metrics = {"auc": 0.90}
        mock_run2.info.run_id = "run2"

        mock_run3 = MagicMock()
        mock_run3.data.metrics = {"auc": 0.82}
        mock_run3.info.run_id = "run3"

        mock_client.search_runs.return_value = [mock_run1, mock_run2, mock_run3]

        manager = MLflowManager()
        manager.experiment_id = "test_exp_id"

        result = manager.get_best_run("auc")

        # Should return run2 with highest AUC
        assert result == mock_run2

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_end_run(self, mock_client, mock_mlflow):
        """Test ending MLflow run"""
        from utils.mlflow_utils import MLflowManager

        manager = MLflowManager()
        manager.end_run("FINISHED")

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("utils.mlflow_utils.mlflow")
    @patch("utils.mlflow_utils.MlflowClient")
    def test_create_model_signature(self, mock_client, mock_mlflow):
        """Test creating model signature"""
        from utils.mlflow_utils import MLflowManager

        mock_signature = MagicMock()
        mock_mlflow.models.infer_signature.return_value = mock_signature

        input_df = pd.DataFrame({"feature1": [1, 2], "feature2": [0.1, 0.2]})
        output_df = pd.DataFrame({"prediction": [0, 1]})

        manager = MLflowManager()
        result = manager.create_model_signature(input_df, output_df)

        mock_mlflow.models.infer_signature.assert_called_once_with(input_df, output_df)
        assert result == mock_signature


class TestMLflowSetup:
    """Test MLflow setup utilities"""

    @patch("utils.mlflow_utils.MLflowManager")
    def test_setup_mlflow_tracking(self, mock_mlflow_manager):
        """Test setting up MLflow tracking"""
        from utils.mlflow_utils import setup_mlflow_tracking

        mock_manager = MagicMock()
        mock_mlflow_manager.return_value = mock_manager

        result = setup_mlflow_tracking(tracking_uri="sqlite:///test.db", experiment_name="test_exp")

        mock_mlflow_manager.assert_called_once_with("sqlite:///test.db", "test_exp")
        assert result == mock_manager
