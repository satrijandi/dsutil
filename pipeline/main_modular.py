#!/usr/bin/env python3
"""
Modular Binary Classification Model Pipeline
Implements complete ML pipeline from data validation to model evaluation
with checkpoints
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from utils.checkpoint import CheckpointManager
# Import utility modules
from utils.data_processing import create_train_test_split, preprocess_data
from utils.feature_engineering import (detect_shifting_features,
                                       feature_selection_lgbm)
from utils.mlflow_utils import setup_mlflow_tracking
from utils.model_utils import evaluate_model, train_h2o_automl

# Configuration constants
DEFAULT_N_FEATURES = 50
DEFAULT_DRIFT_THRESHOLD = 0.3
CONFIG_PATH = "/workspaces/dsutil/pipeline/input/config.yaml"
DATA_PATH = "/workspaces/dsutil/pipeline/input/dataset.parquet"
MLFLOW_DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "ml_pipeline_experiment"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_pipeline_environment() -> tuple[CheckpointManager, object]:
    """Initialize checkpoint manager and MLflow tracking"""
    checkpoint_manager = CheckpointManager()
    
    # Initialize MLflow tracking
    try:
        mlflow_manager = setup_mlflow_tracking(
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="ml_pipeline_experiment",
        )
        logger.info("MLflow tracking initialized")
    except Exception as e:
        logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
        mlflow_manager = None
    
    return checkpoint_manager, mlflow_manager


def start_mlflow_run(mlflow_manager, config: Dict):
    """Start MLflow run with proper tags and parameters"""
    if not mlflow_manager:
        return None
    
    try:
        # Create run tags
        run_tags = {
            "project": config.get("PROJECT_NAME", "ML_Pipeline"),
            "environment": "development",
            "pipeline_version": "modular_v1",
        }

        run_name = f"pipeline_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_run = mlflow_manager.start_run(run_name=run_name, tags=run_tags)

        # Log config parameters
        config_params = {
            "target_col": config.get("TARGET_COL"),
            "date_col": config.get("DATE_COL"),
            "train_start_date": config.get("TRAIN_START_DATE"),
            "oot_start_date": config.get("OOT_START_DATE"),
            "oot_end_date": config.get("OOT_END_DATE"),
        }
        mlflow_manager.log_parameters(config_params)
        return mlflow_run

    except Exception as e:
        logger.warning(f"Failed to start MLflow run: {e}")
        return None


def execute_pipeline_step(checkpoint_manager, step_name: str, step_func, *args, **kwargs):
    """Generic function to execute pipeline steps with checkpointing"""
    result = checkpoint_manager.load_checkpoint(step_name)
    
    if result is None:
        logger.info(f"Executing step: {step_name}")
        result = step_func(*args, **kwargs)
        checkpoint_manager.save_checkpoint(
            step_name,
            result,
            metadata={"step_description": step_name.replace("_", " ").title()}
        )
        logger.info(f"Completed step: {step_name}")
    else:
        logger.info(f"Loaded checkpoint for step: {step_name}")
    
    return result


def finalize_mlflow_run(mlflow_manager, mlflow_run, final_features, shifting_features, train_df, test_df):
    """Finalize MLflow run with summary metrics"""
    if not mlflow_manager or not mlflow_run:
        return
    
    try:
        # Log final pipeline metrics
        pipeline_summary_metrics = {
            "pipeline_success": 1,
            "total_features_selected": len(final_features),
            "features_removed_drift": len(shifting_features),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
        }
        mlflow_manager.log_metrics(pipeline_summary_metrics)

        mlflow_manager.end_run("FINISHED")
        logger.info(f"MLflow run completed: {mlflow_run.info.run_id}")
    except Exception as e:
        logger.warning(f"Failed to end MLflow run properly: {e}")


def main():
    """Main pipeline execution with checkpoints and MLflow tracking"""
    # Configuration
    config_path = "/workspaces/dsutil/pipeline/input/config.yaml"
    data_path = "/workspaces/dsutil/pipeline/input/dataset.parquet"

    # Initialize pipeline environment
    checkpoint_manager, mlflow_manager = setup_pipeline_environment()
    config = load_config(config_path)

    # Start MLflow tracking
    mlflow_run = start_mlflow_run(mlflow_manager, config)

    # Step 1: Load and preprocess data
    step_name = "preprocessed_data"
    df_processed = checkpoint_manager.load_checkpoint(step_name)

    if df_processed is None:
        logger.info("Loading and preprocessing data...")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        df_processed = preprocess_data(df, config)
        checkpoint_manager.save_checkpoint(
            step_name,
            df_processed,
            metadata={
                "shape": df_processed.shape,
                "columns": list(df_processed.columns),
            },
        )
    else:
        logger.info("Using cached preprocessed data")

    # Step 2: Create train/test split
    step_name = "train_test_split"
    split_data = checkpoint_manager.load_checkpoint(step_name)

    if split_data is None:
        logger.info("Creating train/test split...")
        train_df, test_df = create_train_test_split(df_processed, config)

        split_data = {"train_df": train_df, "test_df": test_df}
        checkpoint_manager.save_checkpoint(
            step_name,
            split_data,
            metadata={
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_date_range": [
                    str(train_df[config["DATE_COL"]].min()),
                    str(train_df[config["DATE_COL"]].max()),
                ],
                "test_date_range": (
                    [
                        str(test_df[config["DATE_COL"]].min()),
                        str(test_df[config["DATE_COL"]].max()),
                    ]
                    if len(test_df) > 0
                    else []
                ),
            },
        )
    else:
        logger.info("Using cached train/test split")

    train_df = split_data["train_df"]
    test_df = split_data["test_df"]

    # Step 3: Feature selection
    step_name = "selected_features"
    selected_features = checkpoint_manager.load_checkpoint(step_name)

    if selected_features is None:
        logger.info("Performing feature selection...")
        feature_cols = [
            col
            for col in df_processed.columns
            if col
            not in config["ID_COLUMNS"] + [config["TARGET_COL"], config["DATE_COL"]]
        ]

        X_train = train_df[feature_cols]
        y_train = train_df[config["TARGET_COL"]]

        selected_features = feature_selection_lgbm(X_train, y_train)
        checkpoint_manager.save_checkpoint(
            step_name,
            selected_features,
            metadata={
                "num_features": len(selected_features),
                "features": selected_features[:10],
            },  # Store first 10 for quick reference
        )
    else:
        logger.info("Using cached selected features")

    # Step 4: Detect shifting features
    step_name = "shifting_features"
    shifting_features = checkpoint_manager.load_checkpoint(step_name)

    if shifting_features is None:
        logger.info("Detecting shifting features...")
        shifting_features = detect_shifting_features(train_df, config)
        checkpoint_manager.save_checkpoint(
            step_name,
            shifting_features,
            metadata={
                "num_shifting": len(shifting_features),
                "shifting_features": shifting_features,
            },
        )
    else:
        logger.info("Using cached shifting features detection")

    # Finalize feature list
    final_features = [f for f in selected_features if f not in shifting_features]

    # Safety check: ensure we have at least 2 features for training
    if len(final_features) < 2:
        msg = (
            f"Too few features remaining after drift removal "
            f"({len(final_features)}). Using top 10 selected features instead."
        )
        logger.warning(msg)
        final_features = selected_features[:10]

    msg = (
        f"Final feature count: {len(final_features)} "
        f"(removed {len(shifting_features)} shifting features)"
    )
    logger.info(msg)

    # Step 5: Train model
    step_name = "trained_model"
    aml = checkpoint_manager.load_checkpoint(step_name)

    if aml is None:
        logger.info("Training model...")
        aml = train_h2o_automl(
            train_df, test_df, config, final_features, mlflow_manager
        )
        checkpoint_manager.save_checkpoint(
            step_name,
            aml,
            metadata={
                "model_type": "H2OAutoML",
                "num_features": len(final_features),
            },
        )
    else:
        logger.info("Using cached trained model")

    # Step 6: Evaluate model
    step_name = "evaluation_results"
    evaluation_results = checkpoint_manager.load_checkpoint(step_name)

    if evaluation_results is None:
        logger.info("Evaluating model...")
        evaluation_results = evaluate_model(
            aml, train_df, test_df, config, final_features, mlflow_manager
        )
        checkpoint_manager.save_checkpoint(
            step_name,
            evaluation_results,
            metadata={
                "has_monthly_performance": ("monthly_performance" in evaluation_results)
            },
        )
    else:
        logger.info("Using cached evaluation results")

    # Save results
    output_dir = Path("/workspaces/dsutil/pipeline/output")
    output_dir.mkdir(exist_ok=True)

    # Save monthly performance
    evaluation_results["monthly_performance"].to_csv(
        output_dir / "monthly_performance.csv", index=False
    )

    # Save feature importance
    if evaluation_results["feature_importance"] is not None:
        evaluation_results["feature_importance"].to_csv(
            output_dir / "feature_importance.csv", index=False
        )
    else:
        logger.warning("No feature importance to save")

    # Save model summary
    with open(output_dir / "model_summary.json", "w") as f:
        json.dump(evaluation_results["model_summary"], f, indent=2, default=str)

    # Save pipeline metadata
    pipeline_metadata = {
        "checkpoints_used": checkpoint_manager.list_checkpoints(),
        "final_features": final_features,
        "shifting_features": shifting_features,
        "config": config,
    }

    with open(output_dir / "pipeline_metadata.json", "w") as f:
        json.dump(pipeline_metadata, f, indent=2, default=str)

    # End MLflow run
    if mlflow_manager and mlflow_run:
        try:
            # Log final pipeline metrics
            pipeline_summary_metrics = {
                "pipeline_success": 1,
                "total_features_selected": len(final_features),
                "features_removed_drift": len(shifting_features),
                "training_samples": len(train_df),
                "test_samples": len(test_df),
            }
            mlflow_manager.log_metrics(pipeline_summary_metrics)

            mlflow_manager.end_run("FINISHED")
            logger.info(f"MLflow run completed: {mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run properly: {e}")

    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    msg = f"Checkpoints available at: {checkpoint_manager.checkpoint_dir}"
    logger.info(msg)
    if mlflow_manager:
        logger.info("MLflow tracking data available at: mlflow.db")

    return evaluation_results


if __name__ == "__main__":
    results = main()
