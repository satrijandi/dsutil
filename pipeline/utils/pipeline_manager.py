"""
Pipeline management utilities
"""

import logging

from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages pipeline execution and checkpoint operations"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

    def list_checkpoints(self) -> dict:
        """List all available checkpoints"""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        if not checkpoints:
            print("No checkpoints found.")
            return {}

        print("Available checkpoints:")
        print("-" * 50)
        for step_name, info in checkpoints.items():
            timestamp = info.get("timestamp", "Unknown")
            metadata = info.get("metadata", {})
            print(f"Step: {step_name}")
            print(f"  Saved: {timestamp}")
            if metadata:
                print(f"  Metadata: {metadata}")
            print()

        return checkpoints

    def clear_checkpoint(self, step_name: str) -> bool:
        """Clear a specific checkpoint"""
        if self.checkpoint_manager.clear_checkpoint(step_name):
            print(f"Checkpoint '{step_name}' cleared successfully.")
            return True
        else:
            print(f"Failed to clear checkpoint '{step_name}'.")
            return False

    def clear_all_checkpoints(self) -> bool:
        """Clear all checkpoints"""
        if self.checkpoint_manager.clear_all_checkpoints():
            print("All checkpoints cleared successfully.")
            return True
        else:
            print("Failed to clear all checkpoints.")
            return False

    def restart_from_step(self, step_name: str) -> bool:
        """Restart pipeline from a specific step by clearing subsequent checkpoints"""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        # Define the order of pipeline steps
        step_order = [
            "preprocessed_data",
            "train_test_split",
            "selected_features",
            "shifting_features",
            "trained_model",
            "evaluation_results",
        ]

        try:
            step_index = step_order.index(step_name)
        except ValueError:
            print(f"Unknown step: {step_name}")
            print(f"Available steps: {', '.join(step_order)}")
            return False

        # Clear checkpoints from the specified step onwards
        steps_to_clear = step_order[step_index:]
        cleared_steps = []

        for step in steps_to_clear:
            if step in checkpoints:
                if self.checkpoint_manager.clear_checkpoint(step):
                    cleared_steps.append(step)

        if cleared_steps:
            print(f"Cleared checkpoints for steps: {', '.join(cleared_steps)}")
            print(f"Pipeline will restart from: {step_name}")
        else:
            print(f"No checkpoints to clear. Pipeline will run from: {step_name}")

        return True

    def get_pipeline_status(self) -> dict:
        """Get current pipeline status"""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        step_order = [
            "preprocessed_data",
            "train_test_split",
            "selected_features",
            "shifting_features",
            "trained_model",
            "evaluation_results",
        ]

        status = {}
        for step in step_order:
            status[step] = {
                "completed": step in checkpoints,
                "timestamp": checkpoints.get(step, {}).get("timestamp"),
                "metadata": checkpoints.get(step, {}).get("metadata", {}),
            }

        print("Pipeline Status:")
        print("-" * 50)
        for step, info in status.items():
            status_symbol = "✅" if info["completed"] else "❌"
            timestamp = info["timestamp"] or "Not completed"
            print(f"{status_symbol} {step}: {timestamp}")
            if info["metadata"]:
                print(f"   Metadata: {info['metadata']}")

        return status
