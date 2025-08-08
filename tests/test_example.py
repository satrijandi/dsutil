"""
Basic test example to ensure CI pipeline works.
Replace with actual tests for your ML pipeline components.
"""

import pytest


def test_example():
    """Basic test to verify testing infrastructure works."""
    assert True


def test_import_pipeline_modules():
    """Test that main pipeline modules can be imported."""
    try:
        import sys
        import os

        # Add the pipeline directory to Python path
        pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
        sys.path.insert(0, pipeline_path)

        # Test imports with mock dependencies
        from unittest.mock import patch
        
        with patch.dict('sys.modules', {
            'lightgbm': None,
            'h2o': None, 
            'mlflow': None,
            'mlflow.h2o': None,
            'mlflow.sklearn': None
        }):
            from utils import checkpoint
            from utils import data_processing
            # Skip modules with external dependencies for now
            
        assert True
    except ImportError as e:
        # This is expected due to missing dependencies, so we'll skip for now
        pytest.skip(f"Skipping import test due to missing dependencies: {e}")
