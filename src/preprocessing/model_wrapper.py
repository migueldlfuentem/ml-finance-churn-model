"""
Model wrapper to apply optimized threshold automatically.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper que aplica un threshold optimizado a las predicciones de un clasificador.
    
    Parameters
    ----------
    estimator : estimator
        Clasificador base con método predict_proba
    threshold : float
        Threshold óptimo para clasificación binaria
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> base_model = RandomForestClassifier()
    >>> model = ThresholdClassifier(base_model, threshold=0.53)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)  # Usa threshold 0.53
    """
    
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold
    
    def fit(self, X, y):
        """Entrena el estimador base."""
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self
    
    def predict_proba(self, X):
        """Retorna probabilidades del estimador base."""
        return self.estimator.predict_proba(X)
    
    def predict(self, X):
        """
        Predice usando el threshold optimizado.
        
        Parameters
        ----------
        X : array-like
            Features de entrada
            
        Returns
        -------
        array
            Predicciones binarias usando el threshold optimizado
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def get_params(self, deep=True):
        """Obtiene parámetros del wrapper y del estimador base."""
        params = {
            'estimator': self.estimator,
            'threshold': self.threshold
        }
        if deep and hasattr(self.estimator, 'get_params'):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f'estimator__{k}': v for k, v in estimator_params.items()})
        return params
    
    def set_params(self, **params):
        """Establece parámetros del wrapper y del estimador base."""
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
