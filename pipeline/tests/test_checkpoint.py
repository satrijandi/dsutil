"""
Tests for checkpoint.py module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from utils.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager functionality"""

    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager creates directory and initializes properly"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "test_checkpoints"
            manager = CheckpointManager(str(checkpoint_dir))
            
            # Directory should be created
            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()
            
            # State file should be created if it doesn't exist
            state_file = checkpoint_dir / "pipeline_state.json"
            assert manager.checkpoints_file == state_file
            
            # State should be empty initially
            assert manager.state == {}

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value", "number": 42}
            
            # Save checkpoint
            result = manager.save_checkpoint("test_step", test_data)
            
            # Verify checkpoint file was created
            assert Path(result).exists()
            assert manager.has_checkpoint("test_step")
            
            # Load and verify data
            loaded_data = manager.load_checkpoint("test_step")
            assert loaded_data == test_data

    def test_load_nonexistent_checkpoint(self):
        """Test loading a nonexistent checkpoint returns None"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            
            result = manager.load_checkpoint("nonexistent_step")
            assert result is None

    def test_has_checkpoint(self):
        """Test checking if checkpoint exists"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value"}
            
            # Should not exist initially
            assert not manager.has_checkpoint("test_step")
            
            # Save checkpoint
            manager.save_checkpoint("test_step", test_data)
            
            # Should exist now
            assert manager.has_checkpoint("test_step")

    def test_clear_checkpoint(self):
        """Test clearing a checkpoint"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            test_data = {"key": "value"}
            
            # Save checkpoint
            manager.save_checkpoint("test_step", test_data)
            assert manager.has_checkpoint("test_step")
            
            # Clear checkpoint
            result = manager.clear_checkpoint("test_step")
            assert result is True
            assert not manager.has_checkpoint("test_step")

    def test_dataframe_optimization(self):
        """Test DataFrame optimization during save/load"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(tmp_dir)
            
            # Create test DataFrame
            df = pd.DataFrame({
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0]
            })
            
            # Save and load
            manager.save_checkpoint("df_test", df)
            loaded_df = manager.load_checkpoint("df_test")
            
            # Verify data integrity
            pd.testing.assert_frame_equal(df, loaded_df, check_dtype=False)