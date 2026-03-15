from typing import Dict, Any, Tuple
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_config: Dict[str, Any]):
        self.xgb_params = model_config.get('xgboost', {})
        self.lgb_params = model_config.get('lightgbm', {})
        self.champion_weight = model_config.get('champion_weight', 0.6)
        self.challenger_weight = 1.0 - self.champion_weight
        self.champion_model = xgb.XGBClassifier(**self.xgb_params)
        self.challenger_model = lgb.LGBMClassifier(**self.lgb_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'EnsembleModel':
        self.champion_model.fit(X, y, **fit_params)
        self.challenger_model.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X: np.ndarray, routing: str = 'champion') -> np.ndarray:
        if routing == 'champion':
            return self.champion_model.predict_proba(X)
        elif routing == 'challenger':
            return self.challenger_model.predict_proba(X)
        else:  # blended
            champ_pred = self.champion_model.predict_proba(X) * self.champion_weight
            chall_pred = self.challenger_model.predict_proba(X) * self.challenger_weight
            return champ_pred + chall_pred

    def predict(self, X: np.ndarray, routing: str = 'champion') -> np.ndarray:
        proba = self.predict_proba(X, routing)
        return np.argmax(proba, axis=1)
