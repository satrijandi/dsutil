"""
Comprehensive unit tests for model_serving.py module
"""

import os
import sys
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)


class TestModelServing:
    """Test ModelServing class functionality"""

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_model_serving_initialization_by_stage(self, mock_h2o, mock_load_model):
        """Test ModelServing initialization by stage"""
        from utils.model_serving import ModelServing

        # Mock MLflow manager
        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_mlflow_manager.get_model_version.return_value = mock_version

        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        serving = ModelServing(model_name="test_model", mlflow_manager=mock_mlflow_manager, stage="Production")

        # Verify correct model version retrieval
        mock_mlflow_manager.get_model_version.assert_called_once_with("test_model", stage="Production")

        # Verify model loading with correct URI
        mock_load_model.assert_called_once_with("models:/test_model/Production")

        assert serving.model == mock_model
        assert serving.model_info == mock_version

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_model_serving_initialization_by_version(self, mock_h2o, mock_load_model):
        """Test ModelServing initialization by version"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        serving = ModelServing(model_name="test_model", mlflow_manager=mock_mlflow_manager, version="2")

        mock_mlflow_manager.get_model_version.assert_called_once_with("test_model", version="2")
        mock_load_model.assert_called_once_with("models:/test_model/2")

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_model_serving_initialization_latest(self, mock_h2o, mock_load_model):
        """Test ModelServing initialization with latest version"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "5"
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        serving = ModelServing(model_name="test_model", mlflow_manager=mock_mlflow_manager)

        mock_mlflow_manager.get_model_version.assert_called_once_with("test_model", version="latest")

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_predict_with_dataframe(self, mock_h2o, mock_load_model):
        """Test prediction with DataFrame input"""
        from utils.model_serving import ModelServing

        # Setup mocks
        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock H2O frame and predictions
        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        mock_predictions = MagicMock()
        pred_df = pd.DataFrame({"predict": [0, 1, 0], "p0": [0.7, 0.2, 0.8], "p1": [0.3, 0.8, 0.2]})
        mock_predictions.as_data_frame.return_value = pred_df
        mock_model.predict.return_value = mock_predictions

        serving = ModelServing("test_model", mock_mlflow_manager)

        # Test with DataFrame
        input_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

        result = serving.predict(input_df)

        mock_h2o.H2OFrame.assert_called_once_with(input_df)
        mock_model.predict.assert_called_once_with(mock_h2o_frame)
        pd.testing.assert_frame_equal(result, pred_df)

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_predict_with_dict(self, mock_h2o, mock_load_model):
        """Test prediction with dictionary input"""
        from utils.model_serving import ModelServing

        # Setup mocks
        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        mock_predictions = MagicMock()
        pred_df = pd.DataFrame({"predict": [1], "p1": [0.8]})
        mock_predictions.as_data_frame.return_value = pred_df
        mock_model.predict.return_value = mock_predictions

        serving = ModelServing("test_model", mock_mlflow_manager)

        # Test with dictionary
        input_dict = {"feature1": 1, "feature2": 0.5}
        result = serving.predict(input_dict)

        # Verify dictionary was converted to DataFrame
        expected_df = pd.DataFrame([input_dict])
        mock_h2o.H2OFrame.assert_called_once()
        call_args = mock_h2o.H2OFrame.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, expected_df)

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_predict_with_list_of_dicts(self, mock_h2o, mock_load_model):
        """Test prediction with list of dictionaries"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        mock_predictions = MagicMock()
        pred_df = pd.DataFrame({"predict": [0, 1], "p1": [0.3, 0.8]})
        mock_predictions.as_data_frame.return_value = pred_df
        mock_model.predict.return_value = mock_predictions

        serving = ModelServing("test_model", mock_mlflow_manager)

        # Test with list of dictionaries
        input_list = [{"feature1": 1, "feature2": 0.1}, {"feature1": 2, "feature2": 0.2}]
        result = serving.predict(input_list)

        expected_df = pd.DataFrame(input_list)
        call_args = mock_h2o.H2OFrame.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, expected_df)

    def test_predict_without_model(self):
        """Test prediction when model is not loaded"""
        from utils.model_serving import ModelServing

        with patch("utils.model_serving.mlflow.h2o.load_model") as mock_load:
            mock_load.side_effect = Exception("Load failed")

            mock_mlflow_manager = MagicMock()
            mock_mlflow_manager.get_model_version.side_effect = Exception("Version not found")

            with pytest.raises(Exception):
                ModelServing("test_model", mock_mlflow_manager)

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_predict_proba(self, mock_h2o, mock_load_model):
        """Test getting prediction probabilities"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        # Mock predictions with p1 column
        mock_predictions = MagicMock()
        pred_df = pd.DataFrame({"predict": [0, 1], "p0": [0.7, 0.2], "p1": [0.3, 0.8]})
        mock_predictions.as_data_frame.return_value = pred_df
        mock_model.predict.return_value = mock_predictions

        serving = ModelServing("test_model", mock_mlflow_manager)

        input_df = pd.DataFrame({"feature1": [1, 2]})
        probas = serving.predict_proba(input_df)

        expected_probas = np.array([0.3, 0.8])
        np.testing.assert_array_equal(probas, expected_probas)

    @patch("utils.model_serving.mlflow.h2o.load_model")
    @patch("utils.model_serving.h2o")
    def test_predict_proba_fallback(self, mock_h2o, mock_load_model):
        """Test prediction probabilities with fallback to first column"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_h2o_frame = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_h2o_frame

        # Mock predictions without p1 column
        mock_predictions = MagicMock()
        pred_df = pd.DataFrame({"score": [0.3, 0.8]})
        mock_predictions.as_data_frame.return_value = pred_df
        mock_model.predict.return_value = mock_predictions

        serving = ModelServing("test_model", mock_mlflow_manager)

        input_df = pd.DataFrame({"feature1": [1, 2]})
        probas = serving.predict_proba(input_df)

        expected_probas = np.array([0.3, 0.8])
        np.testing.assert_array_equal(probas, expected_probas)

    @patch("utils.model_serving.mlflow.h2o.load_model")
    def test_get_model_info(self, mock_load_model):
        """Test getting model information"""
        from utils.model_serving import ModelServing

        mock_mlflow_manager = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.description = "Test model"
        mock_version.creation_timestamp = 1234567890
        mock_version.last_updated_timestamp = 1234567900
        mock_version.tags = {"env": "prod"}
        mock_mlflow_manager.get_model_version.return_value = mock_version

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        serving = ModelServing("test_model", mock_mlflow_manager)
        info = serving.get_model_info()

        expected_info = {
            "model_name": "test_model",
            "version": "1",
            "stage": "Production",
            "description": "Test model",
            "creation_timestamp": 1234567890,
            "last_updated_timestamp": 1234567900,
            "tags": {"env": "prod"},
        }

        assert info == expected_info


class TestModelServingAPI:
    """Test ModelServingAPI class functionality"""

    @patch("utils.model_serving.ModelServing")
    def test_model_serving_api_initialization(self, mock_model_serving):
        """Test ModelServingAPI initialization"""
        from utils.model_serving import ModelServingAPI

        mock_serving = MagicMock()
        mock_model_serving.return_value = mock_serving

        mock_mlflow_manager = MagicMock()
        api = ModelServingAPI("test_model", mock_mlflow_manager, version="1", stage="Staging")

        mock_model_serving.assert_called_once_with("test_model", mock_mlflow_manager, "1", "Staging")
        assert api.model_serving == mock_serving
        assert api.model_name == "test_model"

    @patch("utils.model_serving.ModelServing")
    def test_health_check_healthy(self, mock_model_serving):
        """Test health check when model is healthy"""
        from utils.model_serving import ModelServingAPI

        mock_serving = MagicMock()
        mock_serving.get_model_info.return_value = {"version": "1", "status": "ready"}
        mock_model_serving.return_value = mock_serving

        api = ModelServingAPI("test_model", MagicMock())

        with patch("pandas.Timestamp") as mock_timestamp:
            mock_timestamp.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = api.health_check()

        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert result["model_info"] == {"version": "1", "status": "ready"}
        assert result["timestamp"] == "2024-01-01T12:00:00"

    @patch("utils.model_serving.ModelServing")
    def test_health_check_unhealthy(self, mock_model_serving):
        """Test health check when model is unhealthy"""
        from utils.model_serving import ModelServingAPI

        mock_serving = MagicMock()
        mock_serving.get_model_info.side_effect = Exception("Model error")
        mock_model_serving.return_value = mock_serving

        api = ModelServingAPI("test_model", MagicMock())

        with patch("pandas.Timestamp") as mock_timestamp:
            mock_timestamp.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = api.health_check()

        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is False
        assert "error" in result
        assert result["timestamp"] == "2024-01-01T12:00:00"

    @patch("utils.model_serving.ModelServing")
    def test_predict_endpoint_success(self, mock_model_serving):
        """Test successful prediction endpoint"""
        from utils.model_serving import ModelServingAPI

        mock_serving = MagicMock()
        mock_serving.model_info.version = "1"

        # Mock predictions
        pred_df = pd.DataFrame({"predict": [0, 1], "p1": [0.3, 0.8]})
        mock_serving.predict.return_value = pred_df
        mock_serving.predict_proba.return_value = np.array([0.3, 0.8])
        mock_model_serving.return_value = mock_serving

        api = ModelServingAPI("test_model", MagicMock())

        request_data = {"instances": [{"feature1": 1, "feature2": 0.1}, {"feature1": 2, "feature2": 0.2}]}

        with patch("pandas.Timestamp") as mock_timestamp:
            mock_timestamp.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = api.predict_endpoint(request_data)

        assert result["status"] == "success"
        assert "predictions" in result
        assert "probabilities" in result
        assert result["probabilities"] == [0.3, 0.8]
        assert result["model_info"]["version"] == "1"

    @patch("utils.model_serving.ModelServing")
    def test_predict_endpoint_missing_instances(self, mock_model_serving):
        """Test prediction endpoint with missing instances"""
        from utils.model_serving import ModelServingAPI

        api = ModelServingAPI("test_model", MagicMock())

        request_data = {"data": [{"feature1": 1}]}  # Missing 'instances' key
        result = api.predict_endpoint(request_data)

        assert result["status"] == "error"
        assert "Request must contain 'instances' key" in result["error"]

    @patch("utils.model_serving.ModelServing")
    def test_predict_endpoint_prediction_error(self, mock_model_serving):
        """Test prediction endpoint with prediction error"""
        from utils.model_serving import ModelServingAPI

        mock_serving = MagicMock()
        mock_serving.predict.side_effect = Exception("Prediction failed")
        mock_model_serving.return_value = mock_serving

        api = ModelServingAPI("test_model", MagicMock())

        request_data = {"instances": [{"feature1": 1}]}

        with patch("pandas.Timestamp") as mock_timestamp:
            mock_timestamp.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = api.predict_endpoint(request_data)

        assert result["status"] == "error"
        assert "Prediction failed" in result["error"]
        assert result["timestamp"] == "2024-01-01T12:00:00"


class TestModelServingUtilities:
    """Test model serving utility functions"""

    def test_deploy_model_locally(self):
        """Test local model deployment command generation"""
        from utils.model_serving import deploy_model_locally

        mock_mlflow_manager = MagicMock()

        command = deploy_model_locally("test_model", mock_mlflow_manager, port=5555, stage="Staging")

        expected_command = "mlflow models serve -m models:/test_model/Staging -p 5555 --no-conda"
        assert command == expected_command

    def test_create_inference_script(self):
        """Test inference script creation"""
        from utils.model_serving import create_inference_script

        with patch("builtins.open", mock_open()) as mock_file:
            result_path = create_inference_script("test_model", "inference.py")

            assert result_path == "inference.py"
            mock_file.assert_called_once_with("inference.py", "w")

            # Verify script content contains model name
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "test_model" in written_content
            assert "ModelInference" in written_content

    @patch("utils.model_serving.ModelServing")
    def test_batch_predict(self, mock_model_serving):
        """Test batch prediction functionality"""
        from utils.model_serving import batch_predict

        mock_serving = MagicMock()
        pred_df = pd.DataFrame({"predict": [0, 1, 0]})
        probas = np.array([0.3, 0.8, 0.2])
        mock_serving.predict.return_value = pred_df
        mock_serving.predict_proba.return_value = probas
        mock_model_serving.return_value = mock_serving

        input_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

        mock_mlflow_manager = MagicMock()

        result = batch_predict("test_model", input_data, mock_mlflow_manager)

        # Verify result structure
        assert len(result) == 3
        assert "prediction" in result.columns
        assert "probability" in result.columns

        # Verify original features are preserved
        assert "feature1" in result.columns
        assert "feature2" in result.columns

        # Verify predictions and probabilities are added
        np.testing.assert_array_equal(result["probability"].values, probas)

    @patch("utils.model_serving.ModelServing")
    def test_batch_predict_with_output_file(self, mock_model_serving):
        """Test batch prediction with output file"""
        from utils.model_serving import batch_predict

        mock_serving = MagicMock()
        pred_df = pd.DataFrame({"predict": [0, 1]})
        probas = np.array([0.3, 0.8])
        mock_serving.predict.return_value = pred_df
        mock_serving.predict_proba.return_value = probas
        mock_model_serving.return_value = mock_serving

        input_data = pd.DataFrame({"feature1": [1, 2], "feature2": [0.1, 0.2]})

        mock_mlflow_manager = MagicMock()

        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            result = batch_predict("test_model", input_data, mock_mlflow_manager, output_path="output.csv")

            mock_to_csv.assert_called_once_with("output.csv", index=False)
