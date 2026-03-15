from typing import Dict, Any, Tuple
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_config: Dict[str, Any]):
        self.xgb_params = model_config['xgboost']
        self.lgb_params = model_config['lightgbm']
        self.champion_weight = model_config.get('champion_weight', 0.6)
        self.challenger_weight = 1.0 - self.champion_weight
        self.champion_model = xgb.XGBClassifier(**self.xgb_params)
        self.challenger_model = lgb.LGBMClassifier(**self.lgb_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        self.champion_model.fit(X, y)
        self.challenger_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray, use_champion_only: bool = False) -> np.ndarray:
        champ_probs = self.champion_model.predict_proba(X)[:, 1]
        if use_champion_only:
            return champ_probs
        chal_probs = self.challenger_model.predict_proba(X)[:, 1]
        return self.champion_weight * champ_probs + self.challenger_weight * chal_probs

    def predict(self, X: np.ndarray, use_champion_only: bool = False) -> np.ndarray:
        probs = self.predict_proba(X, use_champion_only)
        return (probs >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        champ_imp = self.champion_model.feature_importances_
        chal_imp = self.challenger_model.feature_importances_
        return {
            'champion': champ_imp.tolist(),
            'challenger': chal_imp.tolist()
        }
