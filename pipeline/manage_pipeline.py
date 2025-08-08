#!/usr/bin/env python3
"""
Pipeline management CLI tool
"""

import argparse
import os
import subprocess
import sys

from utils.mlflow_utils import setup_mlflow_tracking
from utils.pipeline_manager import PipelineManager


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Management Tool")
    parser.add_argument(
        "action",
        choices=[
            "status",
            "list",
            "clear",
            "clear-all",
            "restart-from",
            "mlflow-ui",
            "mlflow-experiments",
            "mlflow-models",
            "mlflow-serve",
        ],
        help="Action to perform",
    )
    parser.add_argument("--step", help="Step name for step-specific actions")
    parser.add_argument("--model-name", help="Model name for MLflow operations")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for MLflow UI or serving",
    )
    parser.add_argument("--stage", default="Production", help="Model stage for serving")

    args = parser.parse_args()

    manager = PipelineManager()

    if args.action == "status":
        manager.get_pipeline_status()

    elif args.action == "list":
        manager.list_checkpoints()

    elif args.action == "clear":
        if not args.step:
            print("Error: --step required for clear action")
            sys.exit(1)
        manager.clear_checkpoint(args.step)

    elif args.action == "clear-all":
        confirm = input("Are you sure you want to clear all checkpoints? (y/N): ")
        if confirm.lower() == "y":
            manager.clear_all_checkpoints()
        else:
            print("Operation cancelled.")

    elif args.action == "restart-from":
        if not args.step:
            print("Error: --step required for restart-from action")
            sys.exit(1)
        manager.restart_from_step(args.step)

    # MLflow operations
    elif args.action == "mlflow-ui":
        start_mlflow_ui(args.port)

    elif args.action == "mlflow-experiments":
        show_mlflow_experiments()

    elif args.action == "mlflow-models":
        show_registered_models()

    elif args.action == "mlflow-serve":
        if not args.model_name:
            print("Error: --model-name required for mlflow-serve action")
            sys.exit(1)
        serve_model(args.model_name, args.port, args.stage)


def start_mlflow_ui(port: int = 5000):
    """Start MLflow UI"""
    try:
        print(f"Starting MLflow UI on port {port}...")
        print("MLflow UI will be available at: http://localhost:" + str(port))
        print("Press Ctrl+C to stop the server")

        # Check if mlflow.db exists
        if not os.path.exists("mlflow.db"):
            msg = "Warning: mlflow.db not found. Run the pipeline first to " "create tracking data."
            print(msg)

        # Start MLflow UI
        cmd = f"mlflow ui --backend-store-uri sqlite:///mlflow.db --port {port}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)

    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except Exception as e:
        print(f"Failed to start MLflow UI: {e}")


def show_mlflow_experiments():
    """Show MLflow experiments"""
    try:
        mlflow_manager = setup_mlflow_tracking()
        experiments = mlflow_manager.client.search_experiments()

        if not experiments:
            print("No experiments found.")
            return

        print("MLflow Experiments:")
        print("-" * 60)
        for exp in experiments:
            print(f"Name: {exp.name}")
            print(f"ID: {exp.experiment_id}")
            print(f"Lifecycle Stage: {exp.lifecycle_stage}")
            if exp.tags:
                print(f"Tags: {exp.tags}")

            # Get runs for this experiment
            runs = mlflow_manager.get_experiment_runs(exp.experiment_id, max_results=5)
            print(f"Recent Runs: {len(runs)}")

            for run in runs[:3]:  # Show top 3 runs
                print(f"  - Run ID: {run.info.run_id}")
                print(f"    Status: {run.info.status}")
                print(f"    Start Time: {run.info.start_time}")
                if run.data.metrics:
                    key_metrics = {k: v for k, v in run.data.metrics.items() if k in ["training_auc", "cv_auc_mean"]}
                    if key_metrics:
                        print(f"    Key Metrics: {key_metrics}")
            print()

    except Exception as e:
        print(f"Failed to show experiments: {e}")


def show_registered_models():
    """Show registered models"""
    try:
        mlflow_manager = setup_mlflow_tracking()
        models = mlflow_manager.list_registered_models()

        if not models:
            print("No registered models found.")
            return

        print("Registered Models:")
        print("-" * 60)
        for model in models:
            print(f"Name: {model.name}")
            print(f"Description: {model.description or 'No description'}")
            print(f"Creation Time: {model.creation_timestamp}")

            # Get latest versions
            try:
                versions = mlflow_manager.client.search_model_versions(f"name='{model.name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))
                    print(f"Latest Version: {latest_version.version}")
                    print(f"Current Stage: {latest_version.current_stage}")
                    print(f"Status: {latest_version.status}")

                # Show versions in different stages
                stages = {}
                for version in versions:
                    if version.current_stage not in stages:
                        stages[version.current_stage] = []
                    stages[version.current_stage].append(version.version)

                for stage, vers in stages.items():
                    if stage != "None":
                        print(f"  {stage}: versions {', '.join(vers)}")

            except Exception as e:
                print(f"  Could not get version info: {e}")

            print()

    except Exception as e:
        print(f"Failed to show registered models: {e}")


def serve_model(model_name: str, port: int, stage: str):
    """Serve a model using MLflow"""
    try:
        print(f"Starting model serving for: {model_name}")
        print(f"Stage: {stage}")
        print(f"Port: {port}")
        print(f"Model will be available at: http://localhost:{port}")
        print("Use POST requests to /invocations endpoint for predictions")
        print("Press Ctrl+C to stop the server")

        model_uri = f"models:/{model_name}/{stage}"
        cmd = f"mlflow models serve -m {model_uri} -p {port} --no-conda"
        print(f"Running: {cmd}")

        subprocess.run(cmd, shell=True)

    except KeyboardInterrupt:
        print(f"\nModel serving stopped.")
    except Exception as e:
        print(f"Failed to serve model: {e}")


if __name__ == "__main__":
    main()
