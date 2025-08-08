"""
Comprehensive unit tests for pipeline_manager.py module
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)

from utils.pipeline_manager import PipelineManager


class TestPipelineManager:
    """Test PipelineManager functionality"""

    @patch('utils.pipeline_manager.CheckpointManager')
    def test_pipeline_manager_initialization(self, mock_checkpoint_manager):
        """Test PipelineManager initialization"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        pm = PipelineManager("test_checkpoints")
        
        mock_checkpoint_manager.assert_called_once_with("test_checkpoints")
        assert pm.checkpoint_manager == mock_manager

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_list_checkpoints_with_data(self, mock_print, mock_checkpoint_manager):
        """Test listing checkpoints with data present"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        # Mock checkpoints data
        checkpoints_data = {
            "step1": {
                "timestamp": "2024-01-01T12:00:00",
                "metadata": {"samples": 1000, "features": 50}
            },
            "step2": {
                "timestamp": "2024-01-01T12:30:00",
                "metadata": {}
            }
        }
        mock_manager.list_checkpoints.return_value = checkpoints_data
        
        pm = PipelineManager()
        result = pm.list_checkpoints()
        
        assert result == checkpoints_data
        # Verify print was called for each step
        assert mock_print.call_count > 0

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_list_checkpoints_empty(self, mock_print, mock_checkpoint_manager):
        """Test listing checkpoints when empty"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.list_checkpoints.return_value = {}
        
        pm = PipelineManager()
        result = pm.list_checkpoints()
        
        assert result == {}
        mock_print.assert_called_with("No checkpoints found.")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_clear_checkpoint_success(self, mock_print, mock_checkpoint_manager):
        """Test successfully clearing a checkpoint"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.clear_checkpoint.return_value = True
        
        pm = PipelineManager()
        result = pm.clear_checkpoint("test_step")
        
        assert result is True
        mock_manager.clear_checkpoint.assert_called_once_with("test_step")
        mock_print.assert_called_with("Checkpoint 'test_step' cleared successfully.")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_clear_checkpoint_failure(self, mock_print, mock_checkpoint_manager):
        """Test failing to clear a checkpoint"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.clear_checkpoint.return_value = False
        
        pm = PipelineManager()
        result = pm.clear_checkpoint("test_step")
        
        assert result is False
        mock_print.assert_called_with("Failed to clear checkpoint 'test_step'.")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_clear_all_checkpoints_success(self, mock_print, mock_checkpoint_manager):
        """Test successfully clearing all checkpoints"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.clear_all_checkpoints.return_value = True
        
        pm = PipelineManager()
        result = pm.clear_all_checkpoints()
        
        assert result is True
        mock_manager.clear_all_checkpoints.assert_called_once()
        mock_print.assert_called_with("All checkpoints cleared successfully.")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_clear_all_checkpoints_failure(self, mock_print, mock_checkpoint_manager):
        """Test failing to clear all checkpoints"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.clear_all_checkpoints.return_value = False
        
        pm = PipelineManager()
        result = pm.clear_all_checkpoints()
        
        assert result is False
        mock_print.assert_called_with("Failed to clear all checkpoints.")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_restart_from_step_valid(self, mock_print, mock_checkpoint_manager):
        """Test restarting from a valid step"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        # Mock existing checkpoints
        checkpoints = {
            "preprocessed_data": {"timestamp": "2024-01-01"},
            "train_test_split": {"timestamp": "2024-01-02"},
            "selected_features": {"timestamp": "2024-01-03"},
            "trained_model": {"timestamp": "2024-01-04"}
        }
        mock_manager.list_checkpoints.return_value = checkpoints
        mock_manager.clear_checkpoint.return_value = True
        
        pm = PipelineManager()
        result = pm.restart_from_step("selected_features")
        
        assert result is True
        # Should clear from selected_features onwards
        expected_clears = ["selected_features", "shifting_features", "trained_model", "evaluation_results"]
        actual_clears = [call.args[0] for call in mock_manager.clear_checkpoint.call_args_list]
        
        # Check that appropriate steps were cleared (only those that exist)
        assert "selected_features" in actual_clears
        assert "trained_model" in actual_clears

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_restart_from_step_invalid(self, mock_print, mock_checkpoint_manager):
        """Test restarting from an invalid step"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        pm = PipelineManager()
        result = pm.restart_from_step("invalid_step")
        
        assert result is False
        mock_print.assert_any_call("Unknown step: invalid_step")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_restart_from_step_no_checkpoints_to_clear(self, mock_print, mock_checkpoint_manager):
        """Test restarting from step when no subsequent checkpoints exist"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        # Only has early checkpoints
        checkpoints = {
            "preprocessed_data": {"timestamp": "2024-01-01"},
            "train_test_split": {"timestamp": "2024-01-02"}
        }
        mock_manager.list_checkpoints.return_value = checkpoints
        
        pm = PipelineManager()
        result = pm.restart_from_step("trained_model")
        
        assert result is True
        mock_print.assert_any_call("No checkpoints to clear. Pipeline will run from: trained_model")

    @patch('utils.pipeline_manager.CheckpointManager')
    @patch('builtins.print')
    def test_get_pipeline_status(self, mock_print, mock_checkpoint_manager):
        """Test getting pipeline status"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        checkpoints = {
            "preprocessed_data": {
                "timestamp": "2024-01-01T12:00:00",
                "metadata": {"samples": 1000}
            },
            "train_test_split": {
                "timestamp": "2024-01-01T12:30:00",
                "metadata": {}
            }
        }
        mock_manager.list_checkpoints.return_value = checkpoints
        
        pm = PipelineManager()
        result = pm.get_pipeline_status()
        
        # Verify status structure
        expected_steps = [
            "preprocessed_data", "train_test_split", "selected_features",
            "shifting_features", "trained_model", "evaluation_results"
        ]
        
        for step in expected_steps:
            assert step in result
            assert "completed" in result[step]
            assert "timestamp" in result[step]
            assert "metadata" in result[step]
        
        # Verify completion status
        assert result["preprocessed_data"]["completed"] is True
        assert result["train_test_split"]["completed"] is True
        assert result["selected_features"]["completed"] is False
        assert result["trained_model"]["completed"] is False
        
        # Verify timestamps
        assert result["preprocessed_data"]["timestamp"] == "2024-01-01T12:00:00"
        assert result["selected_features"]["timestamp"] is None
        
        # Verify print was called for status display
        assert mock_print.call_count > 0

    @patch('utils.pipeline_manager.CheckpointManager')
    def test_pipeline_manager_step_order_consistency(self, mock_checkpoint_manager):
        """Test that step order is consistent across methods"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        mock_manager.list_checkpoints.return_value = {}
        
        pm = PipelineManager()
        
        # Get status to check step order
        with patch('builtins.print'):
            status = pm.get_pipeline_status()
        
        expected_order = [
            "preprocessed_data",
            "train_test_split", 
            "selected_features",
            "shifting_features",
            "trained_model",
            "evaluation_results"
        ]
        
        status_keys = list(status.keys())
        assert status_keys == expected_order

    @patch('utils.pipeline_manager.CheckpointManager')
    def test_restart_from_first_step(self, mock_checkpoint_manager):
        """Test restarting from the first step"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        # All steps have checkpoints
        all_steps = [
            "preprocessed_data", "train_test_split", "selected_features", 
            "shifting_features", "trained_model", "evaluation_results"
        ]
        checkpoints = {step: {"timestamp": "2024-01-01"} for step in all_steps}
        mock_manager.list_checkpoints.return_value = checkpoints
        mock_manager.clear_checkpoint.return_value = True
        
        pm = PipelineManager()
        
        with patch('builtins.print'):
            result = pm.restart_from_step("preprocessed_data")
        
        assert result is True
        # Should clear all steps
        assert mock_manager.clear_checkpoint.call_count == len(all_steps)

    @patch('utils.pipeline_manager.CheckpointManager')
    def test_restart_from_last_step(self, mock_checkpoint_manager):
        """Test restarting from the last step"""
        mock_manager = MagicMock()
        mock_checkpoint_manager.return_value = mock_manager
        
        checkpoints = {"evaluation_results": {"timestamp": "2024-01-01"}}
        mock_manager.list_checkpoints.return_value = checkpoints
        mock_manager.clear_checkpoint.return_value = True
        
        pm = PipelineManager()
        
        with patch('builtins.print'):
            result = pm.restart_from_step("evaluation_results")
        
        assert result is True
        # Should only clear the last step
        mock_manager.clear_checkpoint.assert_called_once_with("evaluation_results")