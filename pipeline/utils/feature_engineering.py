"""
Feature selection and drift detection utilities
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import lightgbm as lgb
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def feature_selection_lgbm(
    X: pd.DataFrame, y: pd.Series, n_features: int = 50
) -> List[str]:
    """
    Select features using LightGBM feature importance
    """
    logger.info("Starting feature selection with LightGBM...")

    # Train LightGBM model
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    # Get feature importance
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    selected_features = importance_df.head(n_features)["feature"].tolist()
    logger.info(f"Selected {len(selected_features)} features using LightGBM")

    return selected_features


def _calculate_feature_drift(args: Tuple[str, pd.DataFrame, pd.DataFrame, float]) -> Tuple[str, float, bool]:
    """
    Helper function to calculate drift for a single feature (for parallel processing)
    """
    feature, reference_data, current_data, threshold = args
    
    try:
        ref_values = reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()
        
        if len(ref_values) == 0 or len(cur_values) == 0:
            return feature, 0.0, False
        
        # Use Kolmogorov-Smirnov test
        ks_stat, _ = stats.ks_2samp(ref_values, cur_values)
        is_shifting = ks_stat > threshold
        
        return feature, ks_stat, is_shifting
        
    except Exception:
        return feature, 0.0, False


def detect_shifting_features(
    train_df: pd.DataFrame, config: Dict, threshold: float = 0.3
) -> List[str]:
    """
    Detect shifting features using statistical tests - OPTIMIZED with parallel processing
    """
    logger.info("Detecting shifting features...")

    # Get monthly data starting from TRAIN_START_DATE
    train_start = pd.to_datetime(config["TRAIN_START_DATE"])

    # Create monthly datasets
    train_df_filtered = train_df[train_df[config["DATE_COL"]] >= train_start].copy()
    date_col = config["DATE_COL"]
    train_df_filtered["year_month"] = train_df_filtered[date_col].dt.to_period("M")

    months = sorted(train_df_filtered["year_month"].unique())
    if len(months) < 2:
        logger.warning("Not enough months for drift detection")
        return []

    # Use first month as reference
    reference_month = months[0]
    reference_data = train_df_filtered[
        train_df_filtered["year_month"] == reference_month
    ]

    shifting_features = set()  # Use set for faster lookups
    feature_cols = [
        col
        for col in train_df_filtered.columns
        if col
        not in config["ID_COLUMNS"]
        + [config["TARGET_COL"], config["DATE_COL"], "year_month"]
    ]

    # Process each month in parallel
    for month in months[1:]:
        current_data = train_df_filtered[train_df_filtered["year_month"] == month]

        if len(current_data) == 0 or len(reference_data) == 0:
            continue

        # Prepare arguments for parallel processing
        args_list = [
            (feature, reference_data, current_data, threshold)
            for feature in feature_cols
            if feature not in shifting_features  # Skip already detected features
        ]

        if not args_list:
            continue

        # Use parallel processing for drift calculation
        max_workers = min(4, len(args_list))  # Limit workers to prevent overhead
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_calculate_feature_drift, args_list))

        # Process results
        for feature, ks_stat, is_shifting in results:
            if is_shifting:
                shifting_features.add(feature)
                msg = (
                    f"Shifting feature detected: {feature} "
                    f"(KS stat: {ks_stat:.3f}) in month {month}"
                )
                logger.info(msg)

    shifting_features_list = list(shifting_features)
    logger.info(f"Detected {len(shifting_features_list)} shifting features")
    return shifting_features_list
