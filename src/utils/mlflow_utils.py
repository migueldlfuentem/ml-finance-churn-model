"""
MLflow utility functions for model loading and experiment management.
"""
import mlflow
import mlflow.sklearn
from typing import Optional, Dict
from ..logger import get_logger

logger = get_logger(__name__)


def load_model_from_registry(
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = None
):
    """Load a model from MLflow Model Registry.
    
    Args:
        model_name: Registered model name (e.g., 'churn-catboost').
        version: Specific model version (e.g., '1', '2'). Defaults to None.
        stage: Model stage ('Staging', 'Production', 'Archived'). Defaults to None.
        
    Returns:
        Model loaded from MLflow.
        
    Examples:
        Load specific version:
            >>> model = load_model_from_registry('churn-catboost', version='1')
        
        Load production model:
            >>> model = load_model_from_registry('churn-catboost', stage='Production')
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model: {model_name} version {version}")
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model: {model_name} in stage {stage}")
    else:
        model_uri = f"models:/{model_name}/latest"
        logger.info(f"Loading latest version of: {model_name}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_best_run(
    experiment_name: str,
    metric: str = "f1_mean_cv",
    ascending: bool = False
) -> Dict:
    """Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Experiment name.
        metric: Metric to sort runs. Defaults to "f1_mean_cv".
        ascending: If True, sorts ascending (lower is better). Defaults to False.
        
    Returns:
        Best run information including run_id, metrics, params.
    """
    logger.info(f"Searching for best run in experiment: {experiment_name}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    best_run = runs.iloc[0]
    
    logger.info(f"✓ Best run found:")
    logger.info(f"  Run ID: {best_run['run_id']}")
    logger.info(f"  {metric}: {best_run[f'metrics.{metric}']:.4f}")
    
    return {
        'run_id': best_run['run_id'],
        'metrics': {k.replace('metrics.', ''): v for k, v in best_run.items() if k.startswith('metrics.')},
        'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')},
        'tags': {k.replace('tags.', ''): v for k, v in best_run.items() if k.startswith('tags.')}
    }
