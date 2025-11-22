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
    """
    Carga un modelo desde MLflow Model Registry.
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo registrado (ej: 'churn-catboost')
    version : str, optional
        Versión específica del modelo (ej: '1', '2')
    stage : str, optional
        Stage del modelo ('Staging', 'Production', 'Archived')
        
    Returns
    -------
    model
        Modelo cargado desde MLflow
        
    Examples
    --------
    >>> # Cargar versión específica
    >>> model = load_model_from_registry('churn-catboost', version='1')
    
    >>> # Cargar modelo en producción
    >>> model = load_model_from_registry('churn-catboost', stage='Production')
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Cargando modelo: {model_name} versión {version}")
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Cargando modelo: {model_name} en stage {stage}")
    else:
        # Cargar última versión
        model_uri = f"models:/{model_name}/latest"
        logger.info(f"Cargando última versión de: {model_name}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✓ Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise


def get_best_run(
    experiment_name: str,
    metric: str = "f1_mean_cv",
    ascending: bool = False
) -> Dict:
    """
    Obtiene el mejor run de un experimento basado en una métrica.
    
    Parameters
    ----------
    experiment_name : str
        Nombre del experimento
    metric : str, default="f1_mean_cv"
        Métrica para ordenar runs
    ascending : bool, default=False
        Si True, ordena ascendente (menor es mejor)
        
    Returns
    -------
    Dict
        Información del mejor run incluyendo run_id, metrics, params
    """
    logger.info(f"Buscando mejor run en experimento: {experiment_name}")
    
    # Obtener experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado")
    
    # Buscar runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError(f"No se encontraron runs en el experimento '{experiment_name}'")
    
    best_run = runs.iloc[0]
    
    logger.info(f"✓ Mejor run encontrado:")
    logger.info(f"  Run ID: {best_run['run_id']}")
    logger.info(f"  {metric}: {best_run[f'metrics.{metric}']:.4f}")
    
    return {
        'run_id': best_run['run_id'],
        'metrics': {k.replace('metrics.', ''): v for k, v in best_run.items() if k.startswith('metrics.')},
        'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')},
        'tags': {k.replace('tags.', ''): v for k, v in best_run.items() if k.startswith('tags.')}
    }
