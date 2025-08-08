"""
Comprehensive unit tests for data_processing.py module
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
import sys
import os

# Add the pipeline directory to Python path
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, pipeline_path)

from utils.data_processing import validate_data, preprocess_data, create_train_test_split


class TestValidateData:
    """Test data validation functionality"""

    def test_validate_data_success(self):
        """Test successful data validation"""
        # Create valid test data
        df = pd.DataFrame({
            'id_col': [1, 2, 3, 4, 5],
            'target_col': [0, 1, 0, 1, 1],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'feature1': [1.5, 2.7, 3.2, 4.1, 5.8],
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        result = validate_data(df, config)
        assert result is True

    def test_validate_data_duplicate_rows(self):
        """Test validation fails with duplicate rows"""
        df = pd.DataFrame({
            'id_col': [1, 1, 3],  # Duplicate id with same target
            'target_col': [0, 0, 1],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'feature1': [1.5, 2.7, 3.2]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(ValueError, match="Found .* duplicate rows"):
            validate_data(df, config)

    def test_validate_data_single_class(self):
        """Test validation fails with only one class"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3],
            'target_col': [0, 0, 0],  # Only one class
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'feature1': [1.5, 2.7, 3.2]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(ValueError, match="Only .* unique class"):
            validate_data(df, config)

    def test_validate_data_invalid_date_range(self):
        """Test validation fails with invalid date range"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3],
            'target_col': [0, 1, 0],
            'date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],  # Before train start
            'feature1': [1.5, 2.7, 3.2]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(ValueError, match="Data date range doesn't overlap"):
            validate_data(df, config)

    def test_validate_data_multiple_classes(self):
        """Test validation fails with more than 2 classes"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3, 4],
            'target_col': [0, 1, 2, 0],  # Three classes
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'feature1': [1.5, 2.7, 3.2, 4.1]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(ValueError, match="Target column must have exactly 2 unique values"):
            validate_data(df, config)

    def test_validate_data_invalid_column_names(self):
        """Test validation fails with invalid column names"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3],
            'target_col': [0, 1, 0],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'feature-invalid': [1.5, 2.7, 3.2]  # Invalid column name
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(ValueError, match="Invalid column names"):
            validate_data(df, config)


class TestPreprocessData:
    """Test data preprocessing functionality"""

    def test_preprocess_data_basic(self):
        """Test basic data preprocessing"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3, 4, 5],
            'target_col': [0, 1, 0, 1, 1],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'feature1': [1.567, 2.789, 3.234, 4.123, 5.876],
            'feature2': [10.25, 20.75, np.nan, 40.33, 50.99],
            'ignored_feature': [100, 200, 300, 400, 500]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01',
            'IGNORED_FEATURES': ['ignored_feature']
        }
        
        result = preprocess_data(df, config)
        
        # Check that ignored feature is removed
        assert 'ignored_feature' not in result.columns
        
        # Check that NA values are filled with -999999
        assert result['feature2'].iloc[2] == -999999
        
        # Check that numeric features are rounded to 2 decimal places (rounded up)
        assert result['feature1'].iloc[0] == 1.57  # 1.567 -> 1.57 (rounded up)
        
        # Check that date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date_col'])

    def test_preprocess_data_no_ignored_features(self):
        """Test preprocessing without ignored features config"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3],
            'target_col': [0, 1, 0],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'feature1': [1.5, 2.7, 3.2]
        })
        
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        result = preprocess_data(df, config)
        
        # All columns should be present
        expected_cols = ['id_col', 'target_col', 'date_col', 'feature1']
        assert all(col in result.columns for col in expected_cols)

    def test_preprocess_data_empty_dataframe(self):
        """Test preprocessing with empty dataframe"""
        df = pd.DataFrame()
        config = {
            'ID_COLUMNS': ['id_col'],
            'TARGET_COL': 'target_col',
            'DATE_COL': 'date_col',
            'TRAIN_START_DATE': '2024-01-01'
        }
        
        with pytest.raises(KeyError):
            preprocess_data(df, config)


class TestCreateTrainTestSplit:
    """Test train/test split functionality"""

    def test_create_train_test_split_with_oot_date(self):
        """Test train/test split with explicit OOT date"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3, 4, 5, 6],
            'target_col': [0, 1, 0, 1, 1, 0],
            'date_col': pd.to_datetime([
                '2024-01-01', '2024-01-15', '2024-01-30',
                '2024-02-01', '2024-02-15', '2024-02-28'
            ]),
            'feature1': [1, 2, 3, 4, 5, 6]
        })
        
        config = {
            'DATE_COL': 'date_col',
            'OOT_START_DATE': '2024-02-01'
        }
        
        train_df, test_df = create_train_test_split(df, config)
        
        assert len(train_df) == 3  # Jan data
        assert len(test_df) == 3   # Feb data
        assert train_df['date_col'].max() < pd.to_datetime('2024-02-01')
        assert test_df['date_col'].min() >= pd.to_datetime('2024-02-01')

    def test_create_train_test_split_last_month_logic(self):
        """Test train/test split using last month logic"""
        # Create data for two months
        dates = (
            pd.date_range('2024-01-01', '2024-01-31', freq='D').tolist() +
            pd.date_range('2024-02-01', '2024-02-28', freq='D').tolist()
        )
        
        df = pd.DataFrame({
            'id_col': range(len(dates)),
            'target_col': ([0, 1] * (len(dates) // 2 + 1))[:len(dates)],  # Fix length mismatch
            'date_col': dates,
            'feature1': range(len(dates))
        })
        
        config = {
            'DATE_COL': 'date_col'
        }
        
        train_df, test_df = create_train_test_split(df, config)
        
        # Test set should be February (last month)
        assert all(test_df['date_col'].dt.month == 2)
        # Train set should be January
        assert all(train_df['date_col'].dt.month == 1)

    def test_create_train_test_split_single_month(self):
        """Test train/test split with single month data"""
        df = pd.DataFrame({
            'id_col': [1, 2, 3],
            'target_col': [0, 1, 0],
            'date_col': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-30']),
            'feature1': [1, 2, 3]
        })
        
        config = {
            'DATE_COL': 'date_col'
        }
        
        train_df, test_df = create_train_test_split(df, config)
        
        # With single month, train should be empty and test should have all data
        assert len(train_df) == 0
        assert len(test_df) == 3

    def test_create_train_test_split_empty_dataframe(self):
        """Test train/test split with empty dataframe"""
        df = pd.DataFrame({'date_col': pd.to_datetime([])})  # Empty datetime series
        config = {'DATE_COL': 'date_col'}
        
        train_df, test_df = create_train_test_split(df, config)
        
        assert len(train_df) == 0
        assert len(test_df) == 0