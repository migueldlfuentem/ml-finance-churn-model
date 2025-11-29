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
    """Find the optimal threshold that maximizes the F1-Score.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for the positive class.
        
    Returns:
        Tuple containing (optimal_threshold, best_f1_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
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
    
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    return optimal_threshold, best_f1


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    show_plots: bool = True
) -> Dict[str, float]:
    """Evaluate a trained model and return main metrics.
    
    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: Test target.
        threshold: Threshold for binary classification. Defaults to 0.5.
        show_plots: If True, displays confusion matrix. Defaults to True.
        
    Returns:
        Dictionary with metrics: f1_score, roc_auc, threshold.
    """
    logger.info("Evaluating model...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    if show_plots:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix (Threshold={threshold:.3f})', fontsize=14)
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
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
    """Plot ROC curves for multiple models.
    
    Args:
        results: Dictionary with results per model. Each entry must contain:
            - 'mean_fpr': array with average FPR
            - 'mean_tpr': array with average TPR
            - 'mean_auc': float with average AUC
            - 'std_auc': float with AUC standard deviation
        save_path: Path to save the figure. Defaults to None.
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Random', alpha=0.8)
    
    for model_name, metrics in results.items():
        plt.plot(
            metrics['mean_fpr'], 
            metrics['mean_tpr'],
            label=f"{model_name} (AUC = {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.2f})",
            lw=2, 
            alpha=0.8
        )
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Average ROC Curves (Cross-Validation)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_col: str = 'F1_Optimized',
    save_path: Optional[str] = None
):
    """Plot model comparison based on a metric.
    
    Args:
        results_df: DataFrame with model results (must have 'Model' and metric_col columns).
        metric_col: Name of the column with the metric to compare. Defaults to 'F1_Optimized'.
        save_path: Path to save the figure. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x=metric_col, y='Model', palette='viridis')
    plt.title(f'Model Comparison ({metric_col})', fontsize=14)
    plt.xlim(0, 1)
    plt.xlabel(metric_col, fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to: {save_path}")
    
    plt.show()
