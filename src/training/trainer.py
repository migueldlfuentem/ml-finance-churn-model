"""
Model training utilities with MLflow tracking and cross-validation.
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from mlflow.models.signature import infer_signature
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..evaluation.metrics import find_optimal_threshold
from ..preprocessing.pipelines import create_full_pipeline
from ..preprocessing.model_wrapper import ThresholdClassifier
from ..config import RANDOM_SEED
from ..logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "Bank_Churn_Prediction"
) -> str:
    """
    Configura MLflow tracking.
    
    Parameters
    ----------
    tracking_uri : str, optional
        URI de tracking. Si no se especifica, usa SQLite local.
    experiment_name : str, default="Bank_Churn_Prediction"
        Nombre del experimento
        
    Returns
    -------
    str
        URI de tracking configurada
    """
    if tracking_uri is None:
        db_path = Path.cwd() / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow configurado en: {tracking_uri}")
    logger.info(f"Experimento: {experiment_name}")
    
    return tracking_uri


def train_with_cv(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_splits: int = 5,
    use_smote: bool = True,
    smote_sampling_strategy: float = 0.5,
    log_to_mlflow: bool = True,
    register_model: bool = True
) -> Dict:
    """
    Entrena un modelo con validación cruzada y threshold optimization.
    
    Parameters
    ----------
    model : estimator
        Modelo a entrenar
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Target de entrenamiento
    model_name : str
        Nombre del modelo para logging
    n_splits : int, default=5
        Número de folds para CV
    use_smote : bool, default=True
        Si True, aplica SMOTE
    smote_sampling_strategy : float, default=0.5
        Ratio de sampling para SMOTE
    log_to_mlflow : bool, default=True
        Si True, loguea en MLflow
    register_model : bool, default=True
        Si True, registra el modelo en MLflow Model Registry
        
    Returns
    -------
    Dict
        Diccionario con resultados del entrenamiento:
        - pipeline: Pipeline entrenado
        - f1_mean: F1-Score promedio
        - optimal_threshold: Threshold óptimo promedio
        - mean_auc: AUC promedio
        - std_auc: Desviación estándar del AUC
        - mean_fpr, mean_tpr: Para curvas ROC
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Entrenando: {model_name}")
    logger.info(f"{'='*60}")
    
    pipeline = create_full_pipeline(
        model=model,
        use_smote=use_smote,
        smote_sampling_strategy=smote_sampling_strategy
    )
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    fold_f1s = []
    best_thresholds = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        logger.info(f"  Fold {fold_idx}/{n_splits}...")
        
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        pipeline.fit(X_train_fold, y_train_fold)
        
        y_proba = pipeline.predict_proba(X_val_fold)[:, 1]
        fpr, tpr, _ = roc_curve(y_val_fold, y_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        optimal_thresh, best_f1 = find_optimal_threshold(y_val_fold, y_proba)
        fold_f1s.append(best_f1)
        best_thresholds.append(optimal_thresh)
    
    avg_f1 = np.mean(fold_f1s)
    avg_threshold = np.mean(best_thresholds)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    logger.info(f"  ✓ F1-Score promedio: {avg_f1:.4f}")
    logger.info(f"  ✓ Threshold óptimo: {avg_threshold:.4f}")
    logger.info(f"  ✓ AUC: {mean_auc:.3f} ± {std_auc:.2f}")
    
    logger.info("  Entrenando modelo final con todos los datos y threshold optimizado...")
    pipeline.fit(X_train, y_train)
    
    # Wrappear el pipeline con el threshold optimizado
    optimized_model = ThresholdClassifier(pipeline, threshold=avg_threshold)
    optimized_model.fit(X_train, y_train)  # Fit del wrapper (no re-entrena, solo guarda referencia)
    
    logger.info(f"  ✓ Modelo wrapeado con threshold={avg_threshold:.4f}")
    
    if log_to_mlflow:
        with mlflow.start_run(run_name=f"{model_name} - CV Experiment") as run:
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("experiment_stage", "Cross-Validation")
            mlflow.set_tag("cv_folds", n_splits)
            mlflow.set_tag("uses_optimized_threshold", "True")
            
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("smote_sampling_strategy", smote_sampling_strategy)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("optimized_threshold", avg_threshold)
            
            mlflow.log_metric("f1_mean_cv", avg_f1)
            mlflow.log_metric("optimal_threshold", avg_threshold)
            mlflow.log_metric("auc_mean", mean_auc)
            mlflow.log_metric("auc_std", std_auc)
            
            # Guardar el modelo optimizado (con threshold)
            signature = infer_signature(X_train, optimized_model.predict(X_train))
            
            if register_model:
                model_registry_name = f"churn-{model_name.lower().replace('_', '-')}"
                mlflow.sklearn.log_model(
                    sk_model=optimized_model,
                    name="model",
                    signature=signature,
                    registered_model_name=model_registry_name
                )
                logger.info(f"  ✓ Modelo registrado como: {model_registry_name}")
            else:
                mlflow.sklearn.log_model(
                    sk_model=optimized_model,
                    name="model",
                    signature=signature
                )
    
    results = {
        'pipeline': optimized_model,  # Retornar el modelo optimizado
        'f1_mean': avg_f1,
        'optimal_threshold': avg_threshold,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr
    }
    
    return results


def train_multiple_models(
    models_dict: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    use_smote: bool = True,
    log_to_mlflow: bool = True
) -> pd.DataFrame:
    """
    Entrena múltiples modelos y retorna resultados comparativos.
    
    Parameters
    ----------
    models_dict : Dict
        Diccionario {nombre: instancia_modelo}
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Target de entrenamiento
    n_splits : int, default=5
        Número de folds para CV
    use_smote : bool, default=True
        Si True, aplica SMOTE
    log_to_mlflow : bool, default=True
        Si True, loguea en MLflow
        
    Returns
    -------
    pd.DataFrame
        DataFrame con resultados ordenados por F1-Score
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"INICIANDO ENTRENAMIENTO DE {len(models_dict)} MODELOS")
    logger.info(f"{'#'*60}\n")
    
    results = []
    roc_data = {}
    
    for model_name, model in models_dict.items():
        result = train_with_cv(
            model=model,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            n_splits=n_splits,
            use_smote=use_smote,
            log_to_mlflow=log_to_mlflow
        )
        
        results.append({
            'Model': model_name,
            'F1_Optimized': result['f1_mean'],
            'Threshold': result['optimal_threshold'],
            'AUC': result['mean_auc'],
            'AUC_Std': result['std_auc'],
            'Pipeline': result['pipeline']
        })
        
        roc_data[model_name] = {
            'mean_fpr': result['mean_fpr'],
            'mean_tpr': result['mean_tpr'],
            'mean_auc': result['mean_auc'],
            'std_auc': result['std_auc']
        }
    
    results_df = pd.DataFrame(results).sort_values('F1_Optimized', ascending=False)
    
    logger.info(f"\n{'#'*60}")
    logger.info("RESUMEN DE RESULTADOS")
    logger.info(f"{'#'*60}")
    print(results_df[['Model', 'F1_Optimized', 'Threshold', 'AUC']].to_string(index=False))
    logger.info(f"{'#'*60}\n")
    
    return results_df, roc_data
