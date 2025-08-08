"""
MLflow utilities for model tracking, registry, and serving
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.h2o
import mlflow.sklearn
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow operations for the ML pipeline"""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "ml_pipeline",
    ):
        """
        Initialize MLflow manager

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri)

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment(experiment_name)

        # Set the experiment
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow manager initialized with experiment: {experiment_name}")

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get existing experiment or create new one"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass

        # Create new experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment: {experiment_name} " f"(ID: {experiment_id})")
        return experiment_id

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run

        Returns:
            MLflow active run object
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to MLflow - OPTIMIZED for large parameter sets"""
        # Batch parameter logging for better performance
        if len(params) > 100:
            # Split large parameter sets into smaller batches
            param_items = list(params.items())
            batch_size = 50
            for i in range(0, len(param_items), batch_size):
                batch = dict(param_items[i : i + batch_size])
                mlflow.log_params(batch)
        else:
            mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow - OPTIMIZED for large metric sets"""
        # Filter out invalid metrics to prevent errors
        valid_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not (pd.isna(v) or v == float("inf") or v == float("-inf"))
        }

        if valid_metrics:
            mlflow.log_metrics(valid_metrics, step=step)
            logger.info(f"Logged {len(valid_metrics)} valid metrics to MLflow")

        if len(valid_metrics) != len(metrics):
            logger.warning(f"Filtered out {len(metrics) - len(valid_metrics)} invalid metrics")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
    ):
        """
        Log H2O model to MLflow

        Args:
            model: H2O model to log
            model_name: Name for the model artifact
            signature: Model signature
            input_example: Example input for the model
            registered_model_name: Name to register model under

        Returns:
            Model info object
        """
        try:
            # Log H2O model - OPTIMIZED for memory efficiency
            # Limit input example size to prevent memory issues
            limited_input_example = None
            if input_example is not None:
                limited_input_example = input_example.head(min(5, len(input_example)))

            model_info = mlflow.h2o.log_model(
                h2o_model=model.leader if hasattr(model, "leader") else model,
                artifact_path=model_name,
                signature=signature,
                input_example=limited_input_example,
                registered_model_name=registered_model_name,
            )

            logger.info(f"Logged H2O model: {model_name}")
            if registered_model_name:
                logger.info(f"Registered model as: {registered_model_name}")

            return model_info

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Register a model in MLflow Model Registry

        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Optional description
            tags: Optional tags

        Returns:
            Model version object
        """
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
            except mlflow.exceptions.RestException:
                self.client.create_registered_model(model_name, description=description, tags=tags)
                logger.info(f"Created registered model: {model_name}")

            # Create model version
            model_version = self.client.create_model_version(
                name=model_name, source=model_uri, description=description, tags=tags
            )

            logger.info(f"Registered model version {model_version.version} for {model_name}")
            return model_version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Get a specific model version

        Args:
            model_name: Name of the registered model
            version: Specific version number (or "latest")
            stage: Model stage (e.g., "Production", "Staging")

        Returns:
            Model version object
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model found in stage '{stage}' for {model_name}")
            return versions[0]

        if version is None or version == "latest":
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            return max(versions, key=lambda v: int(v.version))

        return self.client.get_model_version(model_name, version)

    def load_model(self, model_uri: str):
        """Load model from MLflow"""
        try:
            model = mlflow.h2o.load_model(model_uri)
            logger.info(f"Loaded model from: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str, archive_existing: bool = True):
        """
        Transition model to a different stage

        Args:
            model_name: Name of the registered model
            version: Version number to transition
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Whether to archive existing models in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise

    def list_registered_models(
        self,
    ) -> List[mlflow.entities.model_registry.RegisteredModel]:
        """List all registered models"""
        try:
            models = self.client.search_registered_models()
            logger.info(f"Found {len(models)} registered models")
            return models
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            raise

    def get_experiment_runs(self, experiment_id: Optional[str] = None, max_results: int = 1000) -> List[mlflow.entities.Run]:
        """Get runs from an experiment"""
        if experiment_id is None:
            experiment_id = self.experiment_id

        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            logger.info(f"Found {len(runs)} runs in experiment {experiment_id}")
            return runs
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            raise

    def get_best_run(
        self,
        metric_name: str = "auc",
        experiment_id: Optional[str] = None,
        ascending: bool = False,
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric

        Args:
            metric_name: Name of the metric to optimize
            experiment_id: Experiment ID (defaults to current experiment)
            ascending: Whether to sort in ascending order (False for maximizing metrics)

        Returns:
            Best run object or None if no runs found
        """
        runs = self.get_experiment_runs(experiment_id)

        if not runs:
            return None

        # Filter runs that have the specified metric
        runs_with_metric = [run for run in runs if metric_name in run.data.metrics]

        if not runs_with_metric:
            logger.warning(f"No runs found with metric '{metric_name}'")
            return None

        # Sort by metric value
        best_run = sorted(
            runs_with_metric,
            key=lambda r: r.data.metrics[metric_name],
            reverse=not ascending,
        )[0]

        logger.info(f"Best run: {best_run.info.run_id} with {metric_name}={best_run.data.metrics[metric_name]}")
        return best_run

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""
        mlflow.end_run(status=status)
        logger.info("Ended MLflow run")

    def create_model_signature(self, input_example: pd.DataFrame, output_example: Optional[pd.DataFrame] = None):
        """Create model signature from input/output examples"""
        try:
            signature = mlflow.models.infer_signature(input_example, output_example)
            logger.info("Created model signature")
            return signature
        except Exception as e:
            logger.error(f"Failed to create model signature: {e}")
            raise


def setup_mlflow_tracking(tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "ml_pipeline") -> MLflowManager:
    """
    Set up MLflow tracking with local SQLite backend

    Args:
        tracking_uri: MLflow tracking URI (defaults to local SQLite)
        experiment_name: Name of the experiment

    Returns:
        MLflowManager instance
    """
    try:
        manager = MLflowManager(tracking_uri, experiment_name)
        logger.info(f"MLflow tracking setup complete - URI: {tracking_uri}")
        return manager
    except Exception as e:
        logger.error(f"Failed to setup MLflow tracking: {e}")
        raise
