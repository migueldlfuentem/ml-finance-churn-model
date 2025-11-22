from pydantic import BaseModel
from typing import List, Dict, Any

class DataFeaturesSchema(BaseModel):
    numeric: List[str]
    categorical: List[str]

class DataConfigSchema(BaseModel):
    target: str
    features: DataFeaturesSchema

class PreprocessingConfigSchema(BaseModel):
    imputer_strategy_numeric: str
    imputer_strategy_categorical: str

class ModelDefinitionSchema(BaseModel):
    name: str
    params: Dict[str, Any]

class FullConfigSchema(BaseModel):
    data_config: DataConfigSchema
    preprocessing: PreprocessingConfigSchema
    models: Dict[str, ModelDefinitionSchema]


class FeaturesDetail(BaseModel):
    numeric: List[str]
    categorical: List[str]

class SelectedFeaturesConfig(BaseModel):
    comments: str
    date: str
    features: FeaturesDetail