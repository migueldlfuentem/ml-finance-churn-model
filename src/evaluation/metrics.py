"""
Model evaluation utilities including metrics, threshold optimization, and visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc, precision_recall_curve
)
from typing import Tuple, Dict, Optional
from ..logger import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_proba: np.ndarray
) -> Tuple[float, float]:
    """
    Encuentra el threshold óptimo que maximiza el F1-Score.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas
    y_proba : np.ndarray
        Probabilidades predichas para la clase positiva
        
    Returns
    -------
    Tuple[float, float]
        (optimal_threshold, best_f1_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calcular F1 para todos los umbrales
    numerator = 2 * (precisions * recalls)
    denominator = (precisions + recalls)
    f1_scores = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(denominator), 
        where=denominator != 0
    )
    
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"Threshold óptimo encontrado: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    return optimal_threshold, best_f1


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    show_plots: bool = True
) -> Dict[str, float]:
    """
    Evalúa un modelo entrenado y retorna métricas principales.
    
    Parameters
    ----------
    model : estimator
        Modelo entrenado con método predict_proba
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test
    threshold : float, default=0.5
        Threshold para clasificación binaria
    show_plots : bool, default=True
        Si True, muestra matriz de confusión
        
    Returns
    -------
    Dict[str, float]
        Diccionario con métricas: f1_score, roc_auc, threshold
    """
    logger.info("Evaluando modelo...")
    
    # Predicciones
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Métricas
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    # Reporte de clasificación
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    if show_plots:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión (Threshold={threshold:.3f})', fontsize=14)
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.tight_layout()
        plt.show()
    
    metrics = {
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': threshold
    }
    
    return metrics


def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plotea curvas ROC para múltiples modelos.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Diccionario con resultados por modelo. Cada entrada debe contener:
        - 'mean_fpr': array con FPR promedio
        - 'mean_tpr': array con TPR promedio
        - 'mean_auc': float con AUC promedio
        - 'std_auc': float con desviación estándar del AUC
    save_path : str, optional
        Ruta para guardar la figura
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Azar', alpha=0.8)
    
    for model_name, metrics in results.items():
        plt.plot(
            metrics['mean_fpr'], 
            metrics['mean_tpr'],
            label=f"{model_name} (AUC = {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.2f})",
            lw=2, 
            alpha=0.8
        )
    
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curvas ROC Promedio (Validación Cruzada)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curvas ROC guardadas en: {save_path}")
    
    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_col: str = 'F1_Optimized',
    save_path: Optional[str] = None
):
    """
    Plotea comparación de modelos basada en una métrica.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame con resultados de modelos (debe tener columnas 'Model' y metric_col)
    metric_col : str, default='F1_Optimized'
        Nombre de la columna con la métrica a comparar
    save_path : str, optional
        Ruta para guardar la figura
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x=metric_col, y='Model', palette='viridis')
    plt.title(f'Comparativa de Modelos ({metric_col})', fontsize=14)
    plt.xlim(0, 1)
    plt.xlabel(metric_col, fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparación de modelos guardada en: {save_path}")
    
    plt.show()
