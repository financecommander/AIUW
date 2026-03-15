from typing import Dict, Any, Tuple
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_params: Dict[str, Any], lgb_params: Dict[str, Any], champion_weight: float = 0.6):
        self.champion_weight = champion_weight
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.is_champion_xgb = True  # Default champion

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        champion = xgb_probs if self.is_champion_xgb else lgb_probs
        challenger = lgb_probs if self.is_champion_xgb else xgb_probs
        return self.champion_weight * champion + (1 - self.champion_weight) * challenger

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def set_champion(self, is_xgb_champion: bool) -> None:
        self.is_champion_xgb = is_xgb_champion
