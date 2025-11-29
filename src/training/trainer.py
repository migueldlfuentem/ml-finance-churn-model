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
    """Configure MLflow tracking.
    
    Args:
        tracking_uri: Tracking URI. If not specified, uses local SQLite. Defaults to None.
        experiment_name: Experiment name. Defaults to "Bank_Churn_Prediction".
        
    Returns:
        Configured tracking URI.
    """
    if tracking_uri is None:
        db_path = Path.cwd() / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow configured at: {tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")
    
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
    register_model: bool = True,
    use_threshold_wrapper: bool = True
) -> Dict:
    """Train a model with cross-validation and threshold optimization.
    
    Args:
        model: Model to train.
        X_train: Training features.
        y_train: Training target.
        model_name: Model name for logging.
        n_splits: Number of folds for CV. Defaults to 5.
        use_smote: If True, applies SMOTE. Defaults to True.
        smote_sampling_strategy: Sampling ratio for SMOTE. Defaults to 0.5.
        log_to_mlflow: If True, logs to MLflow. Defaults to True.
        register_model: If True, registers model in MLflow Model Registry. Defaults to True.
        use_threshold_wrapper: If True, wraps model with ThresholdClassifier. Defaults to True.
        
    Returns:
        Dictionary with training results:
            - pipeline: Trained pipeline (with or without threshold wrapper)
            - f1_mean: Average F1-Score
            - optimal_threshold: Average optimal threshold
            - mean_auc: Average AUC
            - std_auc: AUC standard deviation
            - mean_fpr, mean_tpr: For ROC curves
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
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
    
    logger.info(f"  ✓ Average F1-Score: {avg_f1:.4f}")
    logger.info(f"  ✓ Optimal threshold: {avg_threshold:.4f}")
    logger.info(f"  ✓ AUC: {mean_auc:.3f} ± {std_auc:.2f}")
    
    logger.info("  Training final model with all data and optimized threshold...")
    pipeline.fit(X_train, y_train)
    
    if use_threshold_wrapper:
        optimized_model = ThresholdClassifier(pipeline, threshold=avg_threshold)
        optimized_model.fit(X_train, y_train)
        logger.info(f"  ✓ Model wrapped with threshold={avg_threshold:.4f}")
    else:
        optimized_model = pipeline
        logger.info(f"  ✓ Model trained without threshold wrapper (using default 0.5)")
    
    if log_to_mlflow:
        with mlflow.start_run(run_name=f"{model_name} - CV Experiment") as run:
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("experiment_stage", "Cross-Validation")
            mlflow.set_tag("cv_folds", n_splits)
            mlflow.set_tag("uses_optimized_threshold", str(use_threshold_wrapper))
            
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("smote_sampling_strategy", smote_sampling_strategy)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("use_threshold_wrapper", use_threshold_wrapper)
            mlflow.log_param("optimized_threshold", avg_threshold)
            
            mlflow.log_metric("f1_mean_cv", avg_f1)
            mlflow.log_metric("optimal_threshold", avg_threshold)
            mlflow.log_metric("auc_mean", mean_auc)
            mlflow.log_metric("auc_std", std_auc)
            
            signature = infer_signature(X_train, optimized_model.predict(X_train))
            
            if register_model:
                model_registry_name = f"churn-{model_name.lower().replace('_', '-')}"
                mlflow.sklearn.log_model(
                    sk_model=optimized_model,
                    name="model",
                    signature=signature,
                    registered_model_name=model_registry_name
                )
                logger.info(f"  ✓ Model registered as: {model_registry_name}")
            else:
                mlflow.sklearn.log_model(
                    sk_model=optimized_model,
                    name="model",
                    signature=signature
                )
    
    results = {
        'pipeline': optimized_model,
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
    log_to_mlflow: bool = True,
    use_threshold_wrapper: bool = True
) -> pd.DataFrame:
    """Train multiple models and return comparative results.
    
    Args:
        models_dict: Dictionary {name: model_instance}.
        X_train: Training features.
        y_train: Training target.
        n_splits: Number of folds for CV. Defaults to 5.
        use_smote: If True, applies SMOTE. Defaults to True.
        log_to_mlflow: If True, logs to MLflow. Defaults to True.
        use_threshold_wrapper: If True, wraps models with ThresholdClassifier. Defaults to True.
        
    Returns:
        DataFrame with results sorted by F1-Score.
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"STARTING TRAINING OF {len(models_dict)} MODELS")
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
            log_to_mlflow=log_to_mlflow,
            use_threshold_wrapper=use_threshold_wrapper
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
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'#'*60}")
    print(results_df[['Model', 'F1_Optimized', 'Threshold', 'AUC']].to_string(index=False))
    logger.info(f"{'#'*60}\n")
    
    return results_df, roc_data
