"""
Comprehensive unit tests for all modules without external dependencies
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)

# Import modules without external dependencies
from utils.checkpoint import CheckpointManager
from utils.data_processing import create_train_test_split, preprocess_data, validate_data
from utils.pipeline_manager import PipelineManager


class TestCheckpointManagerComprehensive:
    """Comprehensive tests for CheckpointManager"""

    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "test_checkpoints"
            manager = CheckpointManager(str(checkpoint_dir))

            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()
            assert manager.checkpoints_file == checkpoint_dir / "pipeline_state.json"
            assert manager.state == {}

    def test_save_and_load_checkpoint_basic(self):
        """Test basic save and load functionality"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value", "number": 42}

            result_path = manager.save_checkpoint("test_step", test_data)
            assert Path(result_path).exists()
            assert manager.has_checkpoint("test_step")

            loaded_data = manager.load_checkpoint("test_step")
            assert loaded_data == test_data

    def test_save_checkpoint_with_metadata(self):
        """Test saving checkpoint with metadata"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value"}
            metadata = {"description": "Test checkpoint", "version": "1.0"}

            manager.save_checkpoint("test_step", test_data, metadata)

            assert "test_step" in manager.state
            assert manager.state["test_step"]["metadata"] == metadata

    def test_dataframe_optimization(self):
        """Test DataFrame memory optimization during save"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)

            df = pd.DataFrame({"int_col": [1, 2, 3, 4, 5], "float_col": [1.1, 2.2, 3.3, 4.4, 5.5]})

            manager.save_checkpoint("df_test", df)
            loaded_df = manager.load_checkpoint("df_test")

            pd.testing.assert_frame_equal(df, loaded_df, check_dtype=False)

    def test_evaluation_results_optimization(self):
        """Test evaluation results optimization during save"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)

            eval_data = {
                "monthly_performance": pd.DataFrame({"month": [1, 2], "auc": [0.8, 0.9]}),
                "model_summary": {"accuracy": 0.85},
                "feature_importance": pd.DataFrame({"feature": ["a", "b"], "importance": [1, 2]}),
                "large_unused_data": {"huge": list(range(1000))},  # Should be filtered out
            }

            manager.save_checkpoint("evaluation_results", eval_data)
            loaded_data = manager.load_checkpoint("evaluation_results")

            assert "monthly_performance" in loaded_data
            assert "model_summary" in loaded_data
            assert "feature_importance" in loaded_data
            assert "large_unused_data" not in loaded_data

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            result = manager.load_checkpoint("nonexistent")
            assert result is None

    def test_clear_checkpoint(self):
        """Test clearing checkpoint"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value"}

            manager.save_checkpoint("test_step", test_data)
            assert manager.has_checkpoint("test_step")

            result = manager.clear_checkpoint("test_step")
            assert result is True
            assert not manager.has_checkpoint("test_step")

    def test_clear_all_checkpoints(self):
        """Test clearing all checkpoints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)

            manager.save_checkpoint("step1", {"data": 1})
            manager.save_checkpoint("step2", {"data": 2})

            result = manager.clear_all_checkpoints()
            assert result is True
            assert len(manager.list_checkpoints()) == 0

    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)

            manager.save_checkpoint("step1", {"data": 1})
            manager.save_checkpoint("step2", {"data": 2})

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            assert "step1" in checkpoints
            assert "step2" in checkpoints

    def test_get_checkpoint_info(self):
        """Test getting checkpoint info"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            metadata = {"description": "Test"}

            manager.save_checkpoint("test_step", {"data": 1}, metadata)

            info = manager.get_checkpoint_info("test_step")
            assert info is not None
            assert info["metadata"] == metadata
            assert "timestamp" in info
            assert "file_size_mb" in info

    def test_compressed_checkpoint_loading(self):
        """Test loading compressed checkpoints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)

            large_data = {"large_list": list(range(1000))}
            manager.save_checkpoint("large_step", large_data)

            loaded_data = manager.load_checkpoint("large_step")
            assert loaded_data == large_data

            # Verify compression flag is set
            assert manager.state["large_step"]["compressed"] is True


class TestDataProcessingComprehensive:
    """Comprehensive tests for data_processing module"""

    def test_validate_data_success(self):
        """Test successful data validation"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3, 4, 5],
                "target_col": [0, 1, 0, 1, 1],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "feature1": [1.5, 2.7, 3.2, 4.1, 5.8],
                "feature_2": [10, 20, 30, 40, 50],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        result = validate_data(df, config)
        assert result is True

    def test_validate_data_duplicate_rows(self):
        """Test validation with duplicate rows"""
        df = pd.DataFrame(
            {
                "id_col": [1, 1, 3],
                "target_col": [0, 0, 1],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "feature1": [1.5, 2.7, 3.2],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        with pytest.raises(ValueError, match="Found .* duplicate rows"):
            validate_data(df, config)

    def test_validate_data_single_class(self):
        """Test validation with only one class"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3],
                "target_col": [0, 0, 0],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "feature1": [1.5, 2.7, 3.2],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        with pytest.raises(ValueError, match="Only .* unique class"):
            validate_data(df, config)

    def test_validate_data_invalid_date_range(self):
        """Test validation with invalid date range"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3],
                "target_col": [0, 1, 0],
                "date_col": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "feature1": [1.5, 2.7, 3.2],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        with pytest.raises(ValueError, match="Data date range doesn't overlap"):
            validate_data(df, config)

    def test_validate_data_multiple_classes(self):
        """Test validation with more than 2 classes"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3, 4],
                "target_col": [0, 1, 2, 0],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "feature1": [1.5, 2.7, 3.2, 4.1],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        with pytest.raises(ValueError, match="Target column must have exactly 2 unique values"):
            validate_data(df, config)

    def test_validate_data_invalid_column_names(self):
        """Test validation with invalid column names"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3],
                "target_col": [0, 1, 0],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "feature-invalid": [1.5, 2.7, 3.2],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
        }

        with pytest.raises(ValueError, match="Invalid column names"):
            validate_data(df, config)

    def test_preprocess_data_basic(self):
        """Test basic data preprocessing"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3, 4, 5],
                "target_col": [0, 1, 0, 1, 1],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "feature1": [1.567, 2.789, 3.234, 4.123, 5.876],
                "feature2": [10.25, 20.75, np.nan, 40.33, 50.99],
                "ignored_feature": [100, 200, 300, 400, 500],
            }
        )

        config = {
            "ID_COLUMNS": ["id_col"],
            "TARGET_COL": "target_col",
            "DATE_COL": "date_col",
            "TRAIN_START_DATE": "2024-01-01",
            "IGNORED_FEATURES": ["ignored_feature"],
        }

        result = preprocess_data(df, config)

        assert "ignored_feature" not in result.columns
        assert result["feature2"].iloc[2] == -999999
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_create_train_test_split_with_oot_date(self):
        """Test train/test split with explicit OOT date"""
        df = pd.DataFrame(
            {
                "id_col": [1, 2, 3, 4, 5, 6],
                "target_col": [0, 1, 0, 1, 1, 0],
                "date_col": pd.to_datetime(
                    ["2024-01-01", "2024-01-15", "2024-01-30", "2024-02-01", "2024-02-15", "2024-02-28"]
                ),
                "feature1": [1, 2, 3, 4, 5, 6],
            }
        )

        config = {"DATE_COL": "date_col", "OOT_START_DATE": "2024-02-01"}

        train_df, test_df = create_train_test_split(df, config)

        assert len(train_df) == 3
        assert len(test_df) == 3
        assert train_df["date_col"].max() < pd.to_datetime("2024-02-01")
        assert test_df["date_col"].min() >= pd.to_datetime("2024-02-01")

    def test_create_train_test_split_empty_dataframe(self):
        """Test train/test split with empty dataframe"""
        df = pd.DataFrame({"date_col": pd.to_datetime([])})
        config = {"DATE_COL": "date_col"}

        train_df, test_df = create_train_test_split(df, config)

        assert len(train_df) == 0
        assert len(test_df) == 0


class TestPipelineManagerComprehensive:
    """Comprehensive tests for PipelineManager"""

    @patch("utils.pipeline_manager.CheckpointManager")
    def test_pipeline_manager_initialization(self, mock_checkpoint_manager):
        """Test PipelineManager initialization"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager

        pm = PipelineManager("test_checkpoints")

        mock_checkpoint_manager.assert_called_once_with("test_checkpoints")
        assert pm.checkpoint_manager == mock_manager

    @patch("utils.pipeline_manager.CheckpointManager")
    @patch("builtins.print")
    def test_list_checkpoints_with_data(self, mock_print, mock_checkpoint_manager):
        """Test listing checkpoints with data present"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager

        checkpoints_data = {
            "step1": {"timestamp": "2024-01-01T12:00:00", "metadata": {"samples": 1000, "features": 50}},
            "step2": {"timestamp": "2024-01-01T12:30:00", "metadata": {}},
        }
        mock_manager.list_checkpoints.return_value = checkpoints_data

        pm = PipelineManager()
        result = pm.list_checkpoints()

        assert result == checkpoints_data
        assert mock_print.call_count > 0

    @patch("utils.pipeline_manager.CheckpointManager")
    @patch("builtins.print")
    def test_clear_checkpoint_success(self, mock_print, mock_checkpoint_manager):
        """Test successfully clearing a checkpoint"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.clear_checkpoint.return_value = True

        pm = PipelineManager()
        result = pm.clear_checkpoint("test_step")

        assert result is True
        mock_manager.clear_checkpoint.assert_called_once_with("test_step")

    @patch("utils.pipeline_manager.CheckpointManager")
    @patch("builtins.print")
    def test_restart_from_step_valid(self, mock_print, mock_checkpoint_manager):
        """Test restarting from a valid step"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager

        checkpoints = {
            "preprocessed_data": {"timestamp": "2024-01-01"},
            "train_test_split": {"timestamp": "2024-01-02"},
            "selected_features": {"timestamp": "2024-01-03"},
            "trained_model": {"timestamp": "2024-01-04"},
        }
        mock_manager.list_checkpoints.return_value = checkpoints
        mock_manager.clear_checkpoint.return_value = True

        pm = PipelineManager()
        result = pm.restart_from_step("selected_features")

        assert result is True
        # Should clear from selected_features onwards
        clear_calls = [call.args[0] for call in mock_manager.clear_checkpoint.call_args_list]
        assert "selected_features" in clear_calls
        assert "trained_model" in clear_calls

    @patch("utils.pipeline_manager.CheckpointManager")
    @patch("builtins.print")
    def test_get_pipeline_status(self, mock_print, mock_checkpoint_manager):
        """Test getting pipeline status"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager

        checkpoints = {
            "preprocessed_data": {"timestamp": "2024-01-01T12:00:00", "metadata": {"samples": 1000}},
            "train_test_split": {"timestamp": "2024-01-01T12:30:00", "metadata": {}},
        }
        mock_manager.list_checkpoints.return_value = checkpoints

        pm = PipelineManager()
        result = pm.get_pipeline_status()

        expected_steps = [
            "preprocessed_data",
            "train_test_split",
            "selected_features",
            "shifting_features",
            "trained_model",
            "evaluation_results",
        ]

        for step in expected_steps:
            assert step in result
            assert "completed" in result[step]
            assert "timestamp" in result[step]
            assert "metadata" in result[step]

        assert result["preprocessed_data"]["completed"] is True
        assert result["train_test_split"]["completed"] is True
        assert result["selected_features"]["completed"] is False
