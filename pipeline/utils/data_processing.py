"""
Data validation and preprocessing utilities
"""

import logging
import re
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, config: Dict) -> bool:
    """
    Validate data according to requirements
    Returns True if valid, raises ValueError if invalid
    """
    logger.info("Starting data validation...")

    # Check if there are duplicate rows based on ID_COLUMNS + target
    duplicate_cols = config["ID_COLUMNS"] + [config["TARGET_COL"]]
    duplicates = df[duplicate_cols].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate rows based on {duplicate_cols}")

    # Check if only one class exists
    unique_labels = df[config["TARGET_COL"]].nunique()
    if unique_labels <= 1:
        raise ValueError(f"Only {unique_labels} unique class(es) found in target column")

    # Check if data falls within specified date range
    df["temp_date"] = pd.to_datetime(df[config["DATE_COL"]])
    train_start = pd.to_datetime(config["TRAIN_START_DATE"])

    min_date = df["temp_date"].min()
    max_date = df["temp_date"].max()
    if min_date > train_start or max_date < train_start:
        raise ValueError("Data date range doesn't overlap with training period")

    # Check if label column has exactly 2 values and is boolean/integer/string
    label_values = df[config["TARGET_COL"]].unique()
    if len(label_values) != 2:
        msg = f"Target column must have exactly 2 unique values, " f"found {len(label_values)}"
        raise ValueError(msg)

    allowed_dtypes = ["int64", "int32", "bool", "object"]
    if not df[config["TARGET_COL"]].dtype in allowed_dtypes:
        msg = f"Target column must be boolean/integer/string, " f"found {df[config['TARGET_COL']].dtype}"
        raise ValueError(msg)

    # Check if all column names contain only alphanumeric characters or underscores
    invalid_cols = [col for col in df.columns if not re.match(r"^[a-zA-Z0-9_]+$", col)]
    if invalid_cols:
        raise ValueError(f"Invalid column names (must be alphanumeric + underscore): {invalid_cols}")

    df.drop("temp_date", axis=1, inplace=True)
    logger.info("Data validation passed")
    return True


def preprocess_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Preprocess data according to requirements
    """
    logger.info("Starting data preprocessing...")

    # Validate data first
    validate_data(df, config)

    # Take only id_columns + [target_col, date_col] - ignored_columns
    required_cols = config["ID_COLUMNS"] + [config["TARGET_COL"], config["DATE_COL"]]
    feature_cols = [col for col in df.columns if col not in required_cols and col not in config.get("IGNORED_FEATURES", [])]

    final_cols = required_cols + feature_cols
    df_processed = df[final_cols].copy()

    # Fill NA with -999999
    df_processed = df_processed.fillna(-999999)

    # Round to 2 decimal places for all numeric features (round up) - OPTIMIZED
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    feature_numeric_cols = [col for col in numeric_cols if col not in required_cols]

    # Vectorized rounding operation for better performance
    if feature_numeric_cols:
        df_processed[feature_numeric_cols] = np.ceil(df_processed[feature_numeric_cols] * 100) / 100

    # Ensure date column is in datetime format
    df_processed[config["DATE_COL"]] = pd.to_datetime(df_processed[config["DATE_COL"]])

    logger.info(f"Data preprocessing completed. Final shape: {df_processed.shape}")
    return df_processed


def create_train_test_split(df: pd.DataFrame, config: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split based on date
    Test is always last month available data unless specified in config
    """
    logger.info("Creating train/test split...")

    df_sorted = df.sort_values(config["DATE_COL"])

    # Use config dates if available, otherwise use last month logic
    if "OOT_START_DATE" in config:
        test_start = pd.to_datetime(config["OOT_START_DATE"])
        train_df = df_sorted[df_sorted[config["DATE_COL"]] < test_start]
        test_df = df_sorted[df_sorted[config["DATE_COL"]] >= test_start]
    else:
        # Last month logic
        if len(df_sorted) == 0:
            train_df = df_sorted.copy()
            test_df = df_sorted.copy()
        else:
            last_date = df_sorted[config["DATE_COL"]].max()
            test_start = last_date.replace(day=1)  # First day of last month
            train_df = df_sorted[df_sorted[config["DATE_COL"]] < test_start]
            test_df = df_sorted[df_sorted[config["DATE_COL"]] >= test_start]

    logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    return train_df, test_df
