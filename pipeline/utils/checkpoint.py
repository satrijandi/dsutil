"""
Checkpoint system to save/load intermediate pipeline results
"""

import json
import logging
import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of pipeline checkpoints"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoints_file = self.checkpoint_dir / "pipeline_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load checkpoint state from file"""
        if self.checkpoints_file.exists():
            with open(self.checkpoints_file, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Save checkpoint state to file"""
        with open(self.checkpoints_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def save_checkpoint(self, step_name: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Save a checkpoint for a pipeline step - OPTIMIZED with compression and selective saving

        Args:
            step_name: Name of the pipeline step
            data: Data to save (DataFrame, dict, etc.)
            metadata: Optional metadata about the checkpoint

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"{step_name}.pkl.xz"

        try:
            # Optimize data before saving based on type
            optimized_data = self._optimize_data_for_storage(data, step_name)

            # Save with compression for better storage efficiency
            with lzma.open(checkpoint_path, "wb", preset=1) as f:  # preset=1 for fast compression
                pickle.dump(optimized_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update state
            file_size = checkpoint_path.stat().st_size
            self.state[step_name] = {
                "file_path": str(checkpoint_path),
                "timestamp": pd.Timestamp.now().isoformat(),
                "metadata": metadata or {},
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "compressed": True,
            }
            self._save_state()

            logger.info(
                f"Checkpoint saved for step '{step_name}' at {checkpoint_path} "
                f"({self.state[step_name]['file_size_mb']} MB)"
            )
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint for step '{step_name}': {e}")
            raise

    def _optimize_data_for_storage(self, data: Any, step_name: str) -> Any:
        """
        Optimize data for storage based on step type - OPTIMIZED for selective saving
        """
        # For DataFrames, optimize memory usage
        if isinstance(data, pd.DataFrame):
            # Create a copy to avoid modifying original
            optimized_df = data.copy()

            # Downcast numeric columns to save space
            for col in optimized_df.select_dtypes(include=["int64"]).columns:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast="integer")

            for col in optimized_df.select_dtypes(include=["float64"]).columns:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast="float")

            return optimized_df

        # For model evaluation results, keep only essential data
        elif isinstance(data, dict) and step_name == "evaluation_results":
            # Keep essential evaluation data, skip large intermediate results
            essential_keys = ["monthly_performance", "model_summary"]
            if "feature_importance" in data and data["feature_importance"] is not None:
                essential_keys.append("feature_importance")

            return {k: v for k, v in data.items() if k in essential_keys}

        # For feature lists, no optimization needed
        elif isinstance(data, list):
            return data

        # For complex objects (like H2O models), save as-is
        else:
            return data

    def load_checkpoint(self, step_name: str) -> Optional[Any]:
        """
        Load a checkpoint for a pipeline step - OPTIMIZED for compressed files

        Args:
            step_name: Name of the pipeline step

        Returns:
            Loaded data or None if checkpoint doesn't exist
        """
        if step_name not in self.state:
            logger.info(f"No checkpoint found for step '{step_name}'")
            return None

        checkpoint_path = Path(self.state[step_name]["file_path"])

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            # Handle both compressed and uncompressed checkpoints for backward compatibility
            is_compressed = self.state[step_name].get("compressed", False)

            if is_compressed or checkpoint_path.suffix == ".xz":
                with lzma.open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)

            timestamp = self.state[step_name]["timestamp"]
            file_size = self.state[step_name].get("file_size_mb", "unknown")
            logger.info(f"Checkpoint loaded for step '{step_name}' (saved: {timestamp}, size: {file_size} MB)")
            return data

        except Exception as e:
            logger.error(f"Failed to load checkpoint for step '{step_name}': {e}")
            return None

    def has_checkpoint(self, step_name: str) -> bool:
        """Check if a checkpoint exists for a step"""
        if step_name not in self.state:
            return False

        checkpoint_path = Path(self.state[step_name]["file_path"])
        return checkpoint_path.exists()

    def clear_checkpoint(self, step_name: str) -> bool:
        """Clear a specific checkpoint"""
        if step_name not in self.state:
            return False

        checkpoint_path = Path(self.state[step_name]["file_path"])

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            del self.state[step_name]
            self._save_state()

            logger.info(f"Checkpoint cleared for step '{step_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to clear checkpoint for step '{step_name}': {e}")
            return False

    def clear_all_checkpoints(self) -> bool:
        """Clear all checkpoints"""
        try:
            for step_name in list(self.state.keys()):
                self.clear_checkpoint(step_name)

            logger.info("All checkpoints cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all checkpoints: {e}")
            return False

    def list_checkpoints(self) -> Dict:
        """List all available checkpoints with metadata"""
        return self.state.copy()

    def get_checkpoint_info(self, step_name: str) -> Optional[Dict]:
        """Get information about a specific checkpoint"""
        return self.state.get(step_name)
