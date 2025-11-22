"""
Pipeline construction utilities for the churn prediction model.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from typing import List, Optional

from ..features.engineering import FeatureEngineer, DropFeatures
from ..config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_SEED
from ..logger import get_logger

logger = get_logger(__name__)


def create_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent'
) -> ColumnTransformer:
    """
    Crea el preprocessor con transformaciones para features numéricas y categóricas.
    
    Parameters
    ----------
    numeric_features : List[str], optional
        Lista de features numéricas. Si no se especifica, usa NUMERIC_FEATURES del config.
    categorical_features : List[str], optional
        Lista de features categóricas. Si no se especifica, usa CATEGORICAL_FEATURES del config.
    numeric_strategy : str, default='median'
        Estrategia de imputación para features numéricas
    categorical_strategy : str, default='most_frequent'
        Estrategia de imputación para features categóricas
        
    Returns
    -------
    ColumnTransformer
        Preprocessor configurado
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    
    logger.info(f"Creando preprocessor con {len(numeric_features)} features numéricas "
                f"y {len(categorical_features)} features categóricas")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def create_full_pipeline(
    model,
    use_smote: bool = True,
    smote_sampling_strategy: float = 0.5,
    features_to_drop: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ImbPipeline:
    """
    Crea el pipeline completo de ML incluyendo feature engineering, preprocessing y modelo.
    
    Parameters
    ----------
    model : estimator
        Modelo de scikit-learn o compatible
    use_smote : bool, default=True
        Si True, aplica SMOTE para balancear clases
    smote_sampling_strategy : float, default=0.5
        Ratio de sampling para SMOTE
    features_to_drop : List[str], optional
        Features a eliminar (ej: ['CustomerId', 'Surname'])
    numeric_features : List[str], optional
        Features numéricas para el preprocessor
    categorical_features : List[str], optional
        Features categóricas para el preprocessor
        
    Returns
    -------
    ImbPipeline
        Pipeline completo configurado
    """
    if features_to_drop is None:
        features_to_drop = ['CustomerId', 'Surname']
    
    logger.info(f"Creando pipeline completo con modelo: {model.__class__.__name__}")
    logger.info(f"SMOTE: {use_smote}, Features a eliminar: {features_to_drop}")
    
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    steps = [
        ('drop_features', DropFeatures(features_to_drop=features_to_drop)),
        ('engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
    ]
    
    if use_smote:
        steps.append(('resampler', SMOTE(random_state=RANDOM_SEED, sampling_strategy=smote_sampling_strategy)))
    
    steps.append(('classifier', model))
    
    pipeline = ImbPipeline(steps=steps)
    
    return pipeline
