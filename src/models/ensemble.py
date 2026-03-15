from typing import Tuple, Dict, Any
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_params: Dict[str, Any], lgb_params: Dict[str, Any], champion_weight: float = 0.7):
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.champion_weight = champion_weight
        self.champion = 'xgb'  # Default champion

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'EnsembleModel':
        self.xgb_model.fit(X, y, **fit_params)
        self.lgb_model.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        if self.champion == 'xgb':
            return self.champion_weight * xgb_probs + (1 - self.champion_weight) * lgb_probs
        return self.champion_weight * lgb_probs + (1 - self.champion_weight) * xgb_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def set_champion(self, champion: str) -> None:
        if champion not in ['xgb', 'lgb']:
            raise ValueError("Champion must be 'xgb' or 'lgb'")
        self.champion = champion
