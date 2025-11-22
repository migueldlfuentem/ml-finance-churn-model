"""
Custom transformers for feature engineering in the churn prediction pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer que crea features derivadas para el modelo de churn.
    
    Features creadas:
    - Balance features: HasBalance, LogBalance, BalancePerProduct, BalanceSalaryRatio
    - Age interactions: Age_Active_Interaction, TenureAgeRatio, CreditScore_Age_Ratio
    - Product features: Products_Active_Interaction, HasMultipleProducts
    - Credit features: CreditScore_Salary_Ratio, HighCreditScore
    - Engagement scores: CustomerEngagement, RiskScore
    
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """No requiere ajuste, retorna self."""
        return self

    def transform(self, X):
        """
        Aplica feature engineering al DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con las features originales
            
        Returns
        -------
        pd.DataFrame
            DataFrame con features originales + features engineered
        """
        X_new = X.copy()

        X_new['HasBalance'] = (X_new['Balance'] > 0).astype(int)
        X_new['LogBalance'] = np.log1p(X_new['Balance'])
        X_new['BalancePerProduct'] = X_new['Balance'] / X_new['NumOfProducts'].replace(0, 1)
        X_new['BalanceSalaryRatio'] = X_new['Balance'] / X_new['EstimatedSalary'].replace(0, 1)

        X_new['Age_Active_Interaction'] = X_new['Age'] * X_new['IsActiveMember']
        X_new['TenureAgeRatio'] = X_new['Tenure'] / X_new['Age'].replace(0, 1)
        X_new['CreditScore_Age_Ratio'] = X_new['CreditScore'] / X_new['Age'].replace(0, 1)

        X_new['Products_Active_Interaction'] = X_new['NumOfProducts'] * X_new['IsActiveMember']
        X_new['HasMultipleProducts'] = (X_new['NumOfProducts'] > 1).astype(int)

        X_new['CreditScore_Salary_Ratio'] = X_new['CreditScore'] / (X_new['EstimatedSalary'] / 1000).replace(0, 1)
        X_new['HighCreditScore'] = (X_new['CreditScore'] > 700).astype(int)

        X_new['CustomerEngagement'] = (
            X_new['IsActiveMember'] + X_new['HasCrCard'] +
            (X_new['NumOfProducts'] > 1).astype(int)
        )

        X_new['RiskScore'] = (
            (X_new['Age'] > 50).astype(int) +
            (X_new['NumOfProducts'] > 2).astype(int) +
            (1 - X_new['IsActiveMember']) +
            (X_new['Balance'] == 0).astype(int)
        )

        return X_new


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer para eliminar columnas que no aportan valor al modelo.
    
    Parameters
    ----------
    features_to_drop : list, optional
        Lista de nombres de columnas a eliminar (ej: ['CustomerId', 'Surname'])
        
    Examples
    --------
    >>> # Uso básico
    >>> drop = DropFeatures(features_to_drop=['CustomerId', 'Surname'])
    >>> X_clean = drop.fit_transform(X)
    
    >>> # En un pipeline
    >>> pipeline = Pipeline([
    ...     ('drop', DropFeatures(['CustomerId', 'Surname'])),
    ...     ('scaler', StandardScaler()),
    ...     ('model', LogisticRegression())
    ... ])
    """
    
    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop if features_to_drop is not None else []

    def fit(self, X, y=None):
        """No requiere ajuste, retorna self."""
        return self

    def transform(self, X):
        """
        Elimina las columnas especificadas si existen en el DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame de entrada
            
        Returns
        -------
        pd.DataFrame
            DataFrame sin las columnas especificadas
        """
        X_out = X.copy()
        cols_to_drop = [col for col in self.features_to_drop if col in X_out.columns]
        
        if cols_to_drop:
            X_out = X_out.drop(columns=cols_to_drop)
        
        return X_out
