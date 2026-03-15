from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class EnsembleModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)
        self.xgb_model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
        return [0.5 * rf + 0.5 * xgb for rf, xgb in zip(rf_probs, xgb_probs)]

    def predict(self, X: List[List[float]]) -> List[int]:
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]
