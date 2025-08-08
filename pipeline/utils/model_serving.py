"""
Model serving utilities for MLflow models
"""

import logging
from typing import Any, Dict, List, Optional, Union

import h2o
import mlflow
import mlflow.h2o
import numpy as np
import pandas as pd

from .mlflow_utils import MLflowManager

logger = logging.getLogger(__name__)


class ModelServing:
    """Model serving interface for MLflow registered models"""

    def __init__(
        self,
        model_name: str,
        mlflow_manager: MLflowManager,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """
        Initialize model serving

        Args:
            model_name: Name of the registered model
            mlflow_manager: MLflow manager instance
            version: Specific version to load (default: latest)
            stage: Model stage to load from (e.g., "Production")
        """
        self.model_name = model_name
        self.mlflow_manager = mlflow_manager
        self.version = version
        self.stage = stage
        self.model = None
        self.model_info = None

        self._load_model()

    def _load_model(self):
        """Load the model from MLflow registry"""
        try:
            # Get model version info
            if self.stage:
                model_version = self.mlflow_manager.get_model_version(self.model_name, stage=self.stage)
                model_uri = f"models:/{self.model_name}/{self.stage}"
            else:
                version = self.version or "latest"
                model_version = self.mlflow_manager.get_model_version(self.model_name, version=version)
                model_uri = f"models:/{self.model_name}/{model_version.version}"

            # Load the model
            self.model = mlflow.h2o.load_model(model_uri)
            self.model_info = model_version

            logger.info(f"Loaded model {self.model_name} v{model_version.version} from MLflow")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def predict(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Make predictions using the loaded model

        Args:
            input_data: Input data for prediction (DataFrame, dict, or list of dicts)

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        try:
            # Convert input to DataFrame if needed
            if isinstance(input_data, dict):
                df_input = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df_input = pd.DataFrame(input_data)
            else:
                df_input = input_data.copy()

            logger.info(f"Making predictions for {len(df_input)} samples")

            # Convert to H2O frame for prediction
            h2o_frame = h2o.H2OFrame(df_input)

            # Make predictions
            predictions = self.model.predict(h2o_frame)
            pred_df = predictions.as_data_frame()

            logger.info("Predictions completed successfully")
            return pred_df

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_proba(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            input_data: Input data for prediction

        Returns:
            Array of prediction probabilities
        """
        predictions = self.predict(input_data)

        # For H2O binary classification, probability is typically in 'p1' column
        if "p1" in predictions.columns:
            return predictions["p1"].values  # type: ignore
        elif len(predictions.columns) == 1:
            return predictions.iloc[:, 0].values  # type: ignore
        else:
            logger.warning("Could not identify probability column. Returning first column.")
            return predictions.iloc[:, 0].values  # type: ignore

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model_info is None:
            return {"error": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "version": self.model_info.version,
            "stage": self.model_info.current_stage,
            "description": self.model_info.description,
            "creation_timestamp": self.model_info.creation_timestamp,
            "last_updated_timestamp": self.model_info.last_updated_timestamp,
            "tags": self.model_info.tags,
        }


class ModelServingAPI:
    """REST API wrapper for model serving"""

    def __init__(
        self,
        model_name: str,
        mlflow_manager: MLflowManager,
        version: Optional[str] = None,
        stage: str = "Production",
    ):
        """
        Initialize serving API

        Args:
            model_name: Name of the registered model
            mlflow_manager: MLflow manager instance
            version: Specific version to serve
            stage: Model stage to serve from
        """
        self.model_serving = ModelServing(model_name, mlflow_manager, version, stage)
        self.model_name = model_name

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            model_info = self.model_serving.get_model_info()
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_info": model_info,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    def predict_endpoint(self, request_data: Dict) -> Dict[str, Any]:
        """
        Prediction endpoint

        Args:
            request_data: Request containing 'instances' key with prediction data

        Returns:
            Response with predictions
        """
        try:
            if "instances" not in request_data:
                return {
                    "error": "Request must contain 'instances' key with prediction data",
                    "status": "error",
                }

            instances = request_data["instances"]
            predictions = self.model_serving.predict(instances)
            probabilities = self.model_serving.predict_proba(instances)

            # Format response
            response = {
                "predictions": predictions.to_dict("records"),
                "probabilities": probabilities.tolist(),
                "model_info": {
                    "name": self.model_name,
                    "version": getattr(self.model_serving.model_info, "version", "unknown"),
                },
                "status": "success",
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            return response

        except Exception as e:
            logger.error(f"Prediction endpoint error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat(),
            }


def deploy_model_locally(
    model_name: str,
    mlflow_manager: MLflowManager,
    port: int = 5001,
    stage: str = "Production",
) -> str:
    """
    Deploy model locally using MLflow's built-in serving

    Args:
        model_name: Name of the registered model
        mlflow_manager: MLflow manager instance
        port: Port to serve on
        stage: Model stage to deploy

    Returns:
        Command to start the server
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"

        # MLflow serve command
        serve_command = f"mlflow models serve -m {model_uri} -p {port} --no-conda"

        logger.info(f"Model deployment command: {serve_command}")
        logger.info(f"Model will be available at: http://localhost:{port}")
        logger.info("Use POST requests to /invocations endpoint for predictions")

        return serve_command

    except Exception as e:
        logger.error(f"Failed to generate deployment command: {e}")
        raise


def create_inference_script(model_name: str, output_path: str = "inference.py") -> str:
    """
    Create a standalone inference script

    Args:
        model_name: Name of the registered model
        output_path: Path to save the inference script

    Returns:
        Path to the created script
    """
    script_content = f'''#!/usr/bin/env python3
"""
Standalone inference script for {model_name}
"""

import mlflow
import mlflow.h2o
import pandas as pd
import h2o
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_uri="models:/{model_name}/Production"):
        """Initialize model for inference"""
        self.model_uri = model_uri
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model from MLflow"""
        try:
            self.model = mlflow.h2o.load_model(self.model_uri)
            logger.info(f"Model loaded from: {{self.model_uri}}")
        except Exception as e:
            logger.error(f"Failed to load model: {{e}}")
            raise
    
    def predict(self, data):
        """Make predictions on input data"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Convert to H2O frame
        h2o_frame = h2o.H2OFrame(df)
        
        # Make predictions
        predictions = self.model.predict(h2o_frame)
        return predictions.as_data_frame()
    
    def predict_proba(self, data):
        """Get prediction probabilities"""
        predictions = self.predict(data)
        if 'p1' in predictions.columns:
            return predictions['p1'].values
        return predictions.iloc[:, 0].values

def main():
    """Example usage"""
    # Initialize inference
    inference = ModelInference()
    
    # Example prediction
    sample_data = {{
        # Add your feature columns here
        "feature1": 1.0,
        "feature2": 2.0
        # ... more features
    }}
    
    predictions = inference.predict(sample_data)
    probabilities = inference.predict_proba(sample_data)
    
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)

if __name__ == "__main__":
    main()
'''

    try:
        with open(output_path, "w") as f:
            f.write(script_content)

        logger.info(f"Inference script created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to create inference script: {e}")
        raise


def batch_predict(
    model_name: str,
    input_data: pd.DataFrame,
    mlflow_manager: MLflowManager,
    output_path: Optional[str] = None,
    stage: str = "Production",
) -> pd.DataFrame:
    """
    Perform batch predictions on large datasets

    Args:
        model_name: Name of the registered model
        input_data: Input DataFrame for predictions
        mlflow_manager: MLflow manager instance
        output_path: Optional path to save predictions
        stage: Model stage to use

    Returns:
        DataFrame with predictions
    """
    try:
        # Initialize model serving
        model_serving = ModelServing(model_name, mlflow_manager, stage=stage)

        logger.info(f"Starting batch prediction for {len(input_data)} samples")

        # Make predictions
        predictions = model_serving.predict(input_data)
        probabilities = model_serving.predict_proba(input_data)

        # Combine with original data
        result_df = input_data.copy()
        result_df["prediction"] = predictions.iloc[:, -1]  # Last column is usually the prediction
        result_df["probability"] = probabilities

        # Save if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to: {output_path}")

        logger.info("Batch prediction completed successfully")
        return result_df

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise
