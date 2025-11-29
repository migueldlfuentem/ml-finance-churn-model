"""
Functions for loading and preparing data.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from ..logger import get_logger
from ..config import RAW_TRAIN_PATH, RAW_TEST_PATH, TARGET_VARIABLE

logger = get_logger(__name__)


def load_train_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the training dataset.
    
    Args:
        path: Path to CSV file. If not specified, uses RAW_TRAIN_PATH from config. Defaults to None.
        
    Returns:
        DataFrame with training data.
    """
    if path is None:
        path = RAW_TRAIN_PATH
    
    logger.info(f"Loading training data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def load_test_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the test dataset.
    
    Args:
        path: Path to CSV file. If not specified, uses RAW_TEST_PATH from config. Defaults to None.
        
    Returns:
        DataFrame with test data.
    """
    if path is None:
        path = RAW_TEST_PATH
    
    logger.info(f"Loading test data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def split_features_target(
    df: pd.DataFrame, 
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target from DataFrame.
    
    Args:
        df: Complete DataFrame with features and target.
        target_col: Target column name. If not specified, uses TARGET_VARIABLE from config. Defaults to None.
        
    Returns:
        Tuple (X, y) where X are features and y is the target.
    """
    if target_col is None:
        target_col = TARGET_VARIABLE
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]} columns")
    logger.info(f"Target: {y.name} (distribution: {y.value_counts().to_dict()})")
    
    return X, y
