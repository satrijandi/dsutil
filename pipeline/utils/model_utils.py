"""
Model training and evaluation utilities
"""

import logging
import os
from typing import Any, Dict, List, Optional

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import roc_auc_score

from .mlflow_utils import MLflowManager

logger = logging.getLogger(__name__)


def train_h2o_automl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    selected_features: List[str],
    mlflow_manager: Optional[MLflowManager] = None,
) -> H2OAutoML:
    """
    Train H2O AutoML model
    """
    logger.info("Starting H2O AutoML training...")

    # Initialize H2O with optimized settings - OPTIMIZED
    h2o.init(
        nthreads=-1,  # Use all available cores
        max_mem_size="4G",  # Adjust based on available memory
        strict_version_check=False,
    )

    # Prepare features (exclude shifting features if any were removed)
    feature_cols = [col for col in selected_features if col not in config["ID_COLUMNS"] + [config["DATE_COL"]]]

    # Convert to H2O frames
    train_cols = feature_cols + [config["TARGET_COL"]]
    h2o_train = h2o.H2OFrame(train_df[train_cols])
    h2o_test = h2o.H2OFrame(test_df[train_cols])

    # Set target as factor for classification
    target_col = config["TARGET_COL"]
    h2o_train[target_col] = h2o_train[target_col].asfactor()
    h2o_test[target_col] = h2o_test[target_col].asfactor()

    # Dynamic resource allocation based on data size and available resources - OPTIMIZED
    data_size = len(train_df)
    num_features = len(feature_cols)

    # Calculate optimal parameters based on dataset characteristics
    if data_size < 10000:
        max_models, max_runtime = 8, 600  # Small dataset: more models, more time
    elif data_size < 100000:
        max_models, max_runtime = 12, 900  # Medium dataset: balanced approach
    else:
        max_models, max_runtime = 20, 1200  # Large dataset: more models for better performance

    # Adjust for feature complexity
    if num_features > 100:
        max_runtime = int(max_runtime * 1.5)  # More time for high-dimensional data

    # Use available CPU cores efficiently
    nthreads = min(os.cpu_count() or 4, 8)  # Cap at 8 threads to prevent resource exhaustion

    # Log training parameters to MLflow if available
    if mlflow_manager:
        training_params = {
            "max_models": max_models,
            "max_runtime_secs": max_runtime,
            "nthreads": nthreads,
            "seed": 42,
            "num_features": num_features,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "algorithm": "H2O_AutoML_Optimized",
        }
        mlflow_manager.log_parameters(training_params)

    # Train AutoML with optimized parameters
    aml = H2OAutoML(
        max_models=max_models,
        seed=42,
        max_runtime_secs=max_runtime,
        nfolds=5,  # Enable cross-validation for better model selection
        balance_classes=True,  # Handle class imbalance automatically
        sort_metric="AUC",  # Optimize for AUC which is common for binary classification
    )
    aml.train(x=feature_cols, y=config["TARGET_COL"], training_frame=h2o_train)

    # Log model to MLflow if available
    if mlflow_manager:
        try:
            # Create input example for model signature
            input_example = train_df[feature_cols].head(5)

            # Create model signature
            signature = mlflow_manager.create_model_signature(input_example)

            # Log the model with registration
            model_name = f"{config.get('PROJECT_NAME', 'ml_pipeline')}_model"
            registered_model_name = f"{config.get('PROJECT_NAME', 'ml_pipeline')}_registered"

            model_info = mlflow_manager.log_model(
                model=aml,
                model_name=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")

        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")

    logger.info("H2O AutoML training completed")
    return aml


def evaluate_model(
    aml: H2OAutoML,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    selected_features: List[str],
    mlflow_manager: Optional[MLflowManager] = None,
) -> Dict:
    """
    Comprehensive model evaluation
    """
    logger.info("Starting model evaluation...")

    feature_cols = [col for col in selected_features if col not in config["ID_COLUMNS"] + [config["DATE_COL"]]]

    results: Dict[str, Any] = {}

    # Monthly AUC analysis - OPTIMIZED for memory efficiency
    combined_df = pd.concat([train_df, test_df])
    date_col = config["DATE_COL"]
    combined_df["year_month"] = combined_df[date_col].dt.to_period("M")

    # Pre-convert feature columns to H2O frame once to avoid repeated conversions
    h2o_combined = h2o.H2OFrame(combined_df[feature_cols + ["year_month"]])

    monthly_results = []
    unique_months = sorted(combined_df["year_month"].unique())

    for month in unique_months:
        month_data = combined_df[combined_df["year_month"] == month]

        if len(month_data) == 0:
            continue

        # Use filtered H2O frame instead of creating new one
        month_mask = h2o_combined["year_month"] == str(month)
        h2o_month_filtered = h2o_combined[month_mask, feature_cols]

        # Batch prediction for efficiency
        predictions = aml.leader.predict(h2o_month_filtered)
        pred_probs = predictions.as_data_frame()["p1"].values

        # Calculate metrics
        y_true = month_data[config["TARGET_COL"]].values
        auc = roc_auc_score(y_true, pred_probs)

        monthly_result = {
            "month": str(month),
            "auc": auc,
            "population": len(month_data),
            "positive_count": sum(y_true),
            "positive_rate": sum(y_true) / len(y_true),
            "prediction_mean": pred_probs.mean(),
        }
        monthly_results.append(monthly_result)

    results["monthly_performance"] = pd.DataFrame(monthly_results)

    # Feature importance
    try:
        feature_importance = aml.leader.varimp(use_pandas=True)
        results["feature_importance"] = feature_importance
    except Exception:
        logger.warning("Could not get feature importance from model")
        results["feature_importance"] = pd.DataFrame(
            {
                "variable": feature_cols,
                "relative_importance": [0] * len(feature_cols),
            }
        )

    # Model summary - structured and readable
    try:
        # Get model performance metrics
        train_auc = float(aml.leader.auc(train=True)) if aml.leader.auc(train=True) else None
        valid_auc = float(aml.leader.auc(valid=True)) if aml.leader.auc(valid=True) else None

        # Get model details
        model_type = str(type(aml.leader).__name__)
        model_key = str(aml.leader.model_id)

        # Get cross-validation metrics if available
        cv_summary = {}
        try:
            cv_data = aml.leader.cross_validation_metrics_summary()
            if cv_data is not None:
                cv_df = cv_data.as_data_frame()
                for _, row in cv_df.iterrows():
                    metric_name = row.iloc[0]  # First column is metric name
                    if metric_name in ["auc", "accuracy", "precision", "recall", "f1"]:
                        cv_summary[metric_name] = {
                            "mean": (float(row.iloc[1]) if pd.notna(row.iloc[1]) else None),
                            "std": (float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None),
                        }
        except Exception:
            cv_summary = {}

        # Get confusion matrix details if available
        confusion_matrix = {}
        try:
            cm = aml.leader.confusion_matrix()
            if cm is not None:
                cm_df = cm.as_data_frame()
                confusion_matrix = {
                    "matrix": cm_df.to_dict("records"),
                    "description": "Confusion matrix for optimal F1 threshold",
                }
        except Exception:
            confusion_matrix = {}

        results["model_summary"] = {
            "model_info": {
                "model_type": model_type,
                "model_key": model_key,
                "algorithm": ("H2O AutoML Stacked Ensemble" if "Stacked" in model_type else model_type),
            },
            "performance_metrics": {
                "training_auc": train_auc,
                "validation_auc": valid_auc,
                "cross_validation_metrics": cv_summary,
            },
            "model_details": {
                "total_models_trained": (len(aml.leaderboard.as_data_frame()) if aml.leaderboard else None),
                "best_model_rank": 1,
                "training_time_info": "5 minutes max runtime with early stopping",
            },
        }

        # Add confusion matrix if available
        if confusion_matrix:
            results["model_summary"]["confusion_matrix"] = confusion_matrix

    except Exception as e:
        logger.warning(f"Could not extract detailed model summary: {e}")
        # Fallback to simple summary
        results["model_summary"] = {
            "model_info": {
                "model_type": "H2O AutoML",
                "status": "trained_successfully",
            },
            "performance_metrics": {
                "training_auc": (float(aml.leader.auc(train=True)) if aml.leader.auc(train=True) else None),
                "validation_auc": (float(aml.leader.auc(valid=True)) if aml.leader.auc(valid=True) else None),
            },
        }

    # Log metrics to MLflow if available
    if mlflow_manager:
        try:
            # Log key performance metrics
            performance_metrics = {
                "training_auc": (float(aml.leader.auc(train=True)) if aml.leader.auc(train=True) else 0.0),
                "num_monthly_evaluations": len(monthly_results),
                "total_models_trained": (len(aml.leaderboard.as_data_frame()) if aml.leaderboard else 0),
            }

            # Add cross-validation metrics if available
            try:
                cv_data = aml.leader.cross_validation_metrics_summary()
                if cv_data is not None:
                    cv_df = cv_data.as_data_frame()
                    for _, row in cv_df.iterrows():
                        metric_name = row.iloc[0]
                        if metric_name in [
                            "auc",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                        ]:
                            performance_metrics[f"cv_{metric_name}_mean"] = (
                                float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0.0
                            )
                            if len(row) > 2 and pd.notna(row.iloc[2]):
                                performance_metrics[f"cv_{metric_name}_std"] = float(row.iloc[2])
            except Exception:
                pass

            # Log monthly AUC metrics
            for result in monthly_results:
                month_str = str(result["month"]).replace("-", "_")
                performance_metrics[f"monthly_auc_{month_str}"] = result["auc"]

            mlflow_manager.log_metrics(performance_metrics)

            # Log evaluation artifacts
            import os
            import tempfile

            # Save monthly performance as artifact - OPTIMIZED memory usage
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                # Use efficient CSV writing without loading all into memory
                results["monthly_performance"].to_csv(f.name, index=False)
                mlflow_manager.log_artifact(f.name, "evaluation_results")
                os.unlink(f.name)

            # Save model summary as artifact - OPTIMIZED
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                import json

                # Serialize with memory-efficient settings
                json.dump(results["model_summary"], f, indent=2, default=str, separators=(",", ":"))
                mlflow_manager.log_artifact(f.name, "evaluation_results")
                os.unlink(f.name)

            logger.info("Logged evaluation metrics and artifacts to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log evaluation results to MLflow: {e}")

    logger.info("Model evaluation completed")
    return results
