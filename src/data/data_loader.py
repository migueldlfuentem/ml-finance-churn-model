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
    """
    Carga el dataset de entrenamiento.
    
    Parameters
    ----------
    path : Path, optional
        Ruta al archivo CSV. Si no se especifica, usa RAW_TRAIN_PATH del config.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos de entrenamiento
    """
    if path is None:
        path = RAW_TRAIN_PATH
    
    logger.info(f"Cargando datos de entrenamiento desde: {path}")
    df = pd.read_csv(path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df


def load_test_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga el dataset de test.
    
    Parameters
    ----------
    path : Path, optional
        Ruta al archivo CSV. Si no se especifica, usa RAW_TEST_PATH del config.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos de test
    """
    if path is None:
        path = RAW_TEST_PATH
    
    logger.info(f"Cargando datos de test desde: {path}")
    df = pd.read_csv(path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df


def split_features_target(
    df: pd.DataFrame, 
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features y target del DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo con features y target
    target_col : str, optional
        Nombre de la columna target. Si no se especifica, usa TARGET_VARIABLE del config.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) donde X son las features y y es el target
    """
    if target_col is None:
        target_col = TARGET_VARIABLE
    
    if target_col not in df.columns:
        raise ValueError(f"Columna target '{target_col}' no encontrada en el DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]} columnas")
    logger.info(f"Target: {y.name} (distribución: {y.value_counts().to_dict()})")
    
    return X, y
