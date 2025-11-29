import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from logger import get_logger
from schema import FullConfigSchema, SelectedFeaturesConfig
from pydantic import ValidationError
from typing import List, Dict, Optional

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

PARAMS_PATH = PROJECT_ROOT / 'config' / 'params.yaml'
SELECTED_FEATURES_PATH = PROJECT_ROOT / 'config' / 'selected_features.json'

def load_params(params_path: Path = PARAMS_PATH) -> FullConfigSchema:
    """Load static configuration from params.yaml validated with Pydantic."""
    logger.info("Loading parameters from %s", params_path)
    if not params_path.exists():
        logger.error("Params file not found: %s", params_path)
        raise FileNotFoundError(f"Params file not found in: {params_path}")
    
    with open(params_path, 'r') as f:
        try:
            params_dict = yaml.safe_load(f)
            if params_dict is None:
                raise ValueError("Empty params.yaml file.")
            return FullConfigSchema(**params_dict)
        except (yaml.YAMLError, ValidationError) as e:
            logger.exception("Error parsing params.yaml")
            raise e

def load_selected_features(json_path: Path = SELECTED_FEATURES_PATH) -> Optional[SelectedFeaturesConfig]:
    """Attempt to load features selected by Boruta.
    
    Returns:
        None if the file doesn't exist (feature selection hasn't been executed yet).
    """
    if not json_path.exists():
        logger.warning(f"{json_path.name} not found. Default features from YAML will be used.")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Dynamic features loaded from {json_path.name}")
            return SelectedFeaturesConfig(**data)
    except json.JSONDecodeError:
        logger.error(f"Error decoding {json_path}. File might be corrupted.")
        return None


params = load_params()

_dynamic_features = load_selected_features()
logger.info(f"Dynamic features: {_dynamic_features}")
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'

RAW_TRAIN_PATH = RAW_DATA_DIR / 'train.csv'
RAW_TEST_PATH = RAW_DATA_DIR / 'test.csv'
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / 'sample_submission.csv'
MODEL_PATH = MODEL_DIR / 'churn_model.joblib'

TARGET_VARIABLE = params.data_config.target

if _dynamic_features:
    NUMERIC_FEATURES = _dynamic_features.features.numeric
    CATEGORICAL_FEATURES = _dynamic_features.features.categorical
    logger.info("🚀 Using DYNAMIC feature configuration (Boruta).")
else:
    NUMERIC_FEATURES = params.data_config.features.numeric
    CATEGORICAL_FEATURES = params.data_config.features.categorical
    logger.info("⚓ Using STATIC feature configuration (params.yaml).")