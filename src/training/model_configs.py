"""
Model configurations for different ML algorithms.
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from ..config import RANDOM_SEED


def get_model_configs():
    """
    Retorna diccionario con configuraciones de modelos predefinidas.
    
    Returns
    -------
    dict
        Diccionario con instancias de modelos configurados
    """
    models = {
        "CatBoost": CatBoostClassifier(
            verbose=0, 
            auto_class_weights='Balanced', 
            random_state=RANDOM_SEED
        ),
        "LGBM": LGBMClassifier(
            is_unbalance=True, 
            random_state=RANDOM_SEED, 
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=4, 
            eval_metric='logloss', 
            random_state=RANDOM_SEED
        ),
        "Random_Forest": RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=150, 
            max_depth=10, 
            random_state=RANDOM_SEED
        ),
        "MLP_Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_SEED
        ),
        "Gaussian_NB": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, 
            weights='distance'
        )
    }
    
    return models
