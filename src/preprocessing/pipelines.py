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
    """Create preprocessor with transformations for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric features. If not specified, uses NUMERIC_FEATURES from config. Defaults to None.
        categorical_features: List of categorical features. If not specified, uses CATEGORICAL_FEATURES from config. Defaults to None.
        numeric_strategy: Imputation strategy for numeric features. Defaults to 'median'.
        categorical_strategy: Imputation strategy for categorical features. Defaults to 'most_frequent'.
        
    Returns:
        Configured ColumnTransformer preprocessor.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    
    logger.info(f"Creating preprocessor with {len(numeric_features)} numeric features "
                f"and {len(categorical_features)} categorical features")
    
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
    numeric_features: Optional[List[str]] = NUMERIC_FEATURES,
    categorical_features: Optional[List[str]] = CATEGORICAL_FEATURES
) -> ImbPipeline:
    """Create complete ML pipeline including feature engineering, preprocessing and model.
    
    Args:
        model: Scikit-learn compatible model.
        use_smote: If True, applies SMOTE for class balancing. Defaults to True.
        smote_sampling_strategy: Sampling ratio for SMOTE. Defaults to 0.5.
        features_to_drop: Features to remove (e.g., ['CustomerId', 'Surname']). Defaults to None.
        numeric_features: Numeric features for preprocessor. Defaults to None.
        categorical_features: Categorical features for preprocessor. Defaults to None.
        
    Returns:
        Configured complete ImbPipeline.
    """
    if features_to_drop is None:
        features_to_drop = [
            'CustomerId',
            'Surname',
        ]
    
    logger.info(f"Creating complete pipeline with model: {model.__class__.__name__}")
    logger.info(f"SMOTE: {use_smote}, Features to drop: {features_to_drop}")

    numeric_features = [f for f in numeric_features if f not in features_to_drop]
    categorical_features = [f for f in categorical_features if f not in features_to_drop]
    
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    steps = [
        ('engineer', FeatureEngineer()),
        ('drop_features', DropFeatures(features_to_drop=features_to_drop)),
        ('preprocessor', preprocessor),
    ]
    
    if use_smote:
        steps.append(('resampler', SMOTE(random_state=RANDOM_SEED, sampling_strategy=smote_sampling_strategy)))
    
    steps.append(('classifier', model))
    
    pipeline = ImbPipeline(steps=steps)
    
    return pipeline
