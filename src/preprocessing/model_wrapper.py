"""
Model wrapper to apply optimized threshold automatically.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that applies an optimized threshold to classifier predictions.
    
    Args:
        estimator: Base classifier with predict_proba method.
        threshold: Optimal threshold for binary classification. Defaults to 0.5.
        
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> base_model = RandomForestClassifier()
        >>> model = ThresholdClassifier(base_model, threshold=0.53)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)  # Uses threshold 0.53
    """
    
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold
    
    def fit(self, X, y):
        """Train the base estimator."""
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self
    
    def predict_proba(self, X):
        """Return probabilities from the base estimator."""
        return self.estimator.predict_proba(X)
    
    def predict(self, X):
        """Predict using the optimized threshold.
        
        Args:
            X: Input features.
            
        Returns:
            Binary predictions using the optimized threshold.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters of the wrapper and base estimator."""
        params = {
            'estimator': self.estimator,
            'threshold': self.threshold
        }
        if deep and hasattr(self.estimator, 'get_params'):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f'estimator__{k}': v for k, v in estimator_params.items()})
        return params
    
    def set_params(self, **params):
        """Set parameters of the wrapper and base estimator."""
        estimator_params = {}
        wrapper_params = {}
        
        for key, value in params.items():
            if key.startswith('estimator__'):
                estimator_params[key.replace('estimator__', '')] = value
            else:
                wrapper_params[key] = value
        
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        
        for key, value in wrapper_params.items():
            setattr(self, key, value)
        
        return self
