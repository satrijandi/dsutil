# ML Pipeline - Modularized Version with MLflow Integration

A robust, modular machine learning pipeline for binary classification with automatic checkpointing, recovery capabilities, and MLflow model registry integration.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with utilities organized in separate modules
- **Automatic Checkpointing**: Pipeline saves intermediate results and can resume from any step
- **Error Recovery**: If something fails, restart from the last successful checkpoint
- **MLflow Integration**: Complete experiment tracking, model registry, and serving capabilities
- **Model Registry**: Automatic model registration with versioning and stage management
- **Pipeline Management**: CLI tools to manage checkpoints, MLflow experiments, and model serving
- **Comprehensive Logging**: Track progress and debug issues easily

## 📁 Structure

```
pipeline/
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── checkpoint.py           # Checkpoint management system
│   ├── data_processing.py      # Data validation and preprocessing
│   ├── feature_engineering.py  # Feature selection and drift detection
│   ├── model_utils.py         # Model training and evaluation
│   ├── mlflow_utils.py        # MLflow tracking and model registry
│   ├── model_serving.py       # Model serving and deployment
│   └── pipeline_manager.py    # Pipeline management utilities
├── main_modular.py            # Main modular pipeline script
├── manage_pipeline.py         # CLI for pipeline management
├── checkpoints/               # Checkpoint storage (created automatically)
├── input/                     # Input data and configuration
├── output/                    # Pipeline results
├── mlflow.db                  # MLflow tracking database (created automatically)
├── mlruns/                    # MLflow artifacts and models (created automatically)
└── README.md                  # This file
```

## 🏃 Quick Start

### 1. Run the Complete Pipeline

```bash
python main_modular.py
```

This will run the entire ML pipeline:
1. Data preprocessing and validation
2. Train/test split creation
3. Feature selection using LightGBM
4. Drift detection and feature filtering
5. H2O AutoML model training
6. Model evaluation and results export

### 2. Check Pipeline Status

```bash
python manage_pipeline.py status
```

See which steps have been completed and their timestamps.

### 3. List All Checkpoints

```bash
python manage_pipeline.py list
```

View detailed information about all saved checkpoints.

### 4. Restart from a Specific Step

```bash
python manage_pipeline.py restart-from --step trained_model
```

Clear checkpoints from a specific step onwards and restart the pipeline from that point.

### 5. Clear Specific Checkpoint

```bash
python manage_pipeline.py clear --step preprocessed_data
```

### 6. Clear All Checkpoints

```bash
python manage_pipeline.py clear-all
```

## 🔬 MLflow Integration

### 7. View MLflow Experiments

```bash
python manage_pipeline.py mlflow-experiments
```

### 8. View Registered Models

```bash
python manage_pipeline.py mlflow-models
```

### 9. Start MLflow UI

```bash
python manage_pipeline.py mlflow-ui --port 5000
```

Open http://localhost:5000 in your browser to explore experiments and models.

### 10. Serve a Model

```bash
python manage_pipeline.py mlflow-serve --model-name DummyData_registered --port 5001
```

## 📊 Pipeline Steps

The pipeline consists of these sequential steps with full MLflow tracking:

1. **preprocessed_data**: Load and preprocess raw data
2. **train_test_split**: Create temporal train/test split
3. **selected_features**: Feature selection using LightGBM importance
4. **shifting_features**: Detect and remove drifting features
5. **trained_model**: Train H2O AutoML model + log to MLflow + register model
6. **evaluation_results**: Evaluate model and generate metrics + log metrics to MLflow

### 🔬 MLflow Tracking Features

- **Automatic Experiment Tracking**: Every pipeline run creates an MLflow experiment
- **Parameter Logging**: Training parameters, data sizes, feature counts automatically logged
- **Metrics Logging**: AUC scores, cross-validation metrics, monthly performance tracked
- **Model Registry**: Trained models automatically registered with versioning
- **Artifact Storage**: Evaluation results, model summaries saved as artifacts
- **Model Serving**: Deploy registered models with single CLI command

## 🔧 Configuration

Edit `input/config.yaml` to customize:

```yaml
PROJECT_NAME: "YourProject"
TARGET_COL: "your_target_column"
ID_COLUMNS: ["your_id_column"]
DATE_COL: "your_date_column"
TRAIN_START_DATE: "2024-01-01"
OOT_START_DATE: "2024-09-01"
OOT_END_DATE: "2024-09-30"
IGNORED_FEATURES: ["feature_to_ignore"]
```

## 📈 Output Files

The pipeline generates:

- `monthly_performance.csv`: Monthly AUC and performance metrics
- `feature_importance.csv`: Feature importance scores (if available)
- `model_summary.json`: Detailed model information and statistics
- `pipeline_metadata.json`: Pipeline execution metadata and configuration

## 🚀 Model Serving & Deployment

### Serve Models Locally

Use the built-in MLflow serving:

```bash
python manage_pipeline.py mlflow-serve --model-name DummyData_registered --port 5001
```

### Make Predictions

Once serving, send POST requests to `http://localhost:5001/invocations`:

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feat_1": 1.0, "feat_2": 2.0, "feat_3": 3.0}]}'
```

### Programmatic Model Loading

```python
from utils.model_serving import ModelServing
from utils.mlflow_utils import setup_mlflow_tracking

# Setup MLflow
mlflow_manager = setup_mlflow_tracking()

# Load model
model_serving = ModelServing("DummyData_registered", mlflow_manager)

# Make predictions
predictions = model_serving.predict(your_data_df)
probabilities = model_serving.predict_proba(your_data_df)
```

## 🔄 Checkpoint System

### How It Works

The checkpoint system automatically saves intermediate results after each step:

- **Automatic Saving**: Results are pickled and saved with metadata
- **Smart Loading**: Pipeline checks for existing checkpoints before running each step
- **Metadata Tracking**: Each checkpoint includes timestamp and relevant metadata
- **Error Recovery**: If a step fails, previous steps don't need to re-run

### Checkpoint Files

Checkpoints are stored in the `checkpoints/` directory:

- `pipeline_state.json`: Tracks all checkpoint metadata
- `{step_name}.pkl`: Binary pickle files containing the actual data

### Example Recovery Scenario

1. Pipeline fails during model training
2. Check status: `python manage_pipeline.py status`
3. Restart from feature selection: `python manage_pipeline.py restart-from --step selected_features`
4. Re-run pipeline: `python main_modular.py`

## 🛠️ Utility Modules

### data_processing.py
- Data validation (duplicates, date ranges, column types)
- Preprocessing (missing value handling, rounding, type conversion)
- Train/test splitting based on dates

### feature_engineering.py
- LightGBM-based feature selection
- Statistical drift detection using Kolmogorov-Smirnov test
- Feature importance ranking

### model_utils.py
- H2O AutoML model training
- Comprehensive model evaluation
- Monthly performance analysis

### checkpoint.py
- Checkpoint creation and loading
- Metadata management
- State persistence

### pipeline_manager.py
- CLI interface for checkpoint management
- Pipeline status tracking
- Restart capabilities

## 🚨 Error Handling

The modular pipeline includes robust error handling:

- **Validation Errors**: Clear error messages for data validation failures
- **Checkpoint Failures**: Graceful degradation when checkpoints can't be loaded
- **Model Training Issues**: Safety checks for feature availability
- **Recovery Options**: Multiple ways to restart and recover from failures

## 🔍 Debugging

### Check Logs
All modules use Python logging. Increase verbosity if needed:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Checkpoints
Load and examine checkpoint data:

```python
from utils.checkpoint import CheckpointManager
manager = CheckpointManager()
data = manager.load_checkpoint("preprocessed_data")
```

### Pipeline Status
Always check status before debugging:

```bash
python manage_pipeline.py status
```

## 🎯 Benefits

1. **Reliability**: Automatic recovery from failures
2. **Efficiency**: Skip completed steps when re-running
3. **Debugging**: Easy to isolate and fix issues in specific steps
4. **Maintainability**: Clean, modular code structure
5. **Flexibility**: Easy to modify or extend individual components
6. **Reproducibility**: Full MLflow experiment tracking for reproducible results
7. **Model Management**: Automated model registry with versioning and stages
8. **Easy Deployment**: One-command model serving and deployment
9. **Experiment Comparison**: Compare runs and models through MLflow UI
10. **Production Ready**: Industry-standard MLflow for model lifecycle management

## 📝 Notes

- The original `main.py` was removed - use `main_modular.py` for the complete pipeline
- H2O models can't be perfectly checkpointed due to internal constraints, but will be retrained quickly if needed
- MLflow automatically tracks all experiments, models, and artifacts
- Checkpoint files can grow large with big datasets - monitor disk space
- MLflow database (`mlflow.db`) stores all experiment metadata
- Clear old checkpoints periodically to free up space
- Models are automatically registered in MLflow with versioning
- Use MLflow UI to compare experiments and manage model lifecycle
- Model serving is production-ready with REST API interface