"""
Evaluation module for model assessment, metrics and visualization.
"""
from .metrics import (
    evaluate_model,
    find_optimal_threshold,
    plot_roc_curves,
    plot_model_comparison
)

__all__ = [
    'evaluate_model',
    'find_optimal_threshold',
    'plot_roc_curves',
    'plot_model_comparison'
]
